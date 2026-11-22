"""TARK Studio — Multi-Provider Vector Embeddings System.

Provides 768-dimensional semantic dense vector embeddings for RAG:
1. Ollama (default for local development, model: nomic-embed-text, 768d)
2. Cloudflare Workers AI (production, model: @cf/google/embeddinggemma-2, 768d)

Architecture guarantees:
- Local development is 100% preserved (Ollama + nomic-embed-text).
- Production is 100% independent of local machine via Cloudflare Workers AI HTTPS REST API.
- Unified 768-dimensional vector space across both providers for pgvector compatibility.
"""

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
from collections import OrderedDict
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768
DEFAULT_BATCH_SIZE = 32
CLOUDFLARE_DEFAULT_BATCH_SIZE = 4
QUERY_CACHE_MAX_SIZE = 256


class _EmbeddingLRUCache:
    """Thread-safe LRU cache for query embeddings to eliminate repeated inference latency."""

    def __init__(self, maxsize: int = QUERY_CACHE_MAX_SIZE):
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._maxsize = maxsize
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[List[float]]:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    async def put(self, key: str, value: List[float]) -> None:
        async with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)


class BaseEmbeddingProvider(ABC):
    """Abstract base class for vector embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the active embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension produced by this provider."""
        pass

    @abstractmethod
    async def get_embedding(self, text: str, is_query: bool = True) -> List[float]:
        """Generate embedding vector for a single text or query."""
        pass

    @abstractmethod
    async def get_embeddings_batch(
        self, texts: List[str], is_query: bool = False
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts in batches."""
        pass

    async def ensure_model_available(self) -> bool:
        """Verify provider availability on startup."""
        return True


class OllamaEmbeddings(BaseEmbeddingProvider):
    """Asynchronous client for Ollama embedding generation (Local Development)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.embedding_model or "nomic-embed-text"
        self.batch_size = batch_size
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    async def get_embedding(self, text: str, is_query: bool = True) -> List[float]:
        """Generate embedding vector for a single query or text passage."""
        results = await self.get_embeddings_batch([text], is_query=is_query)
        if not results:
            raise RuntimeError(f"Ollama failed to return an embedding for prompt.")
        return results[0]

    async def get_embeddings_batch(
        self, texts: List[str], is_query: bool = False
    ) -> List[List[float]]:
        """Generate embeddings in batches using Ollama /api/embed or /api/embeddings."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # 1. Try batch /api/embed endpoint
                try:
                    payload = {"model": self.model, "input": batch}
                    res = await client.post(f"{self.base_url}/api/embed", json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        embeddings = data.get("embeddings", [])
                        if len(embeddings) == len(batch):
                            all_embeddings.extend(embeddings)
                            continue
                except Exception as e:
                    logger.debug(f"/api/embed batch call fell back: {e}")

                # 2. Fallback to individual /api/embeddings endpoint
                for t in batch:
                    payload = {"model": self.model, "prompt": t}
                    res = await client.post(f"{self.base_url}/api/embeddings", json=payload)
                    if res.status_code != 200:
                        raise RuntimeError(
                            f"Ollama embedding request failed with HTTP {res.status_code}: {res.text}"
                        )
                    emb = res.json().get("embedding")
                    if not emb or len(emb) != EMBEDDING_DIM:
                        raise ValueError(
                            f"Expected {EMBEDDING_DIM}d embedding from {self.model}, got {len(emb) if emb else 0}d"
                        )
                    all_embeddings.append(emb)

        return all_embeddings

    async def ensure_model_available(self) -> bool:
        """Verify that the embedding model is available in Ollama, pulling it if absent."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    models = [m.get("name", "") for m in res.json().get("models", [])]
                    if any(self.model in name for name in models):
                        logger.info(f"Ollama model '{self.model}' is verified available.")
                        return True
                    logger.info(f"Ollama model '{self.model}' not found in {models}. Pulling model...")
                    pull_res = await client.post(
                        f"{self.base_url}/api/pull",
                        json={"name": self.model, "stream": False},
                        timeout=300.0,
                    )
                    if pull_res.status_code == 200:
                        logger.info(f"Ollama model '{self.model}' pulled successfully.")
                        return True
                    logger.warning(f"Ollama pull returned status {pull_res.status_code}: {pull_res.text}")
        except Exception as exc:
            logger.warning(f"Could not verify/pull Ollama model '{self.model}': {exc}")
        return False


# Global query embedding cache shared across requests
_query_cache = _EmbeddingLRUCache(QUERY_CACHE_MAX_SIZE)


class CloudflareEmbeddings(BaseEmbeddingProvider):
    """Cloudflare Workers AI client for EmbeddingGemma-2 (768d, Production).

    Calls Cloudflare Workers AI HTTPS REST API endpoint directly.
    Completely independent of local machines, tunnels, or Ollama daemons.
    Features:
    - Persistent HTTP connection pooling with keep-alive
    - Bounded concurrent batch execution (asyncio.Semaphore)
    - Thread-safe LRU caching for query embeddings
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        model: str = "@cf/google/embeddinggemma-2",
        batch_size: int = CLOUDFLARE_DEFAULT_BATCH_SIZE,
        timeout: float = 60.0,
    ):
        self.account_id = (account_id or settings.cloudflare_account_id or "").strip()
        self.api_token = (api_token or settings.cloudflare_api_token or "").strip()
        self.model = (model or settings.cloudflare_embedding_model or "@cf/google/embeddinggemma-2").strip()
        self.batch_size = batch_size
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        if not self.account_id or not self.api_token:
            logger.warning(
                "CloudflareEmbeddings initialized without complete credentials. "
                "Ensure CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are configured in production."
            )

    def _get_client(self) -> httpx.AsyncClient:
        """Get or initialize a pooled, persistent AsyncClient with keep-alive."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=15.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0),
            )
        return self._client

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    def _format_input(self, text: str, is_query: bool = False) -> str:
        """Format text with Google EmbeddingGemma task instructions if appropriate."""
        cleaned = text.strip()
        if is_query:
            if not cleaned.lower().startswith("task:"):
                return f"task: search result | query: {cleaned}"
        return cleaned

    async def get_embedding(self, text: str, is_query: bool = True) -> List[float]:
        """Generate 768d vector for a single query or text passage with LRU caching for queries."""
        cleaned = text.strip()
        if is_query:
            cache_key = f"cf:{self.model}:{cleaned}"
            cached = await _query_cache.get(cache_key)
            if cached is not None:
                return cached

        results = await self.get_embeddings_batch([text], is_query=is_query)
        if not results:
            raise RuntimeError(f"Cloudflare Workers AI returned no embeddings for model '{self.model}'.")
        emb = results[0]

        if is_query:
            cache_key = f"cf:{self.model}:{cleaned}"
            await _query_cache.put(cache_key, emb)

        return emb

    async def get_embeddings_batch(
        self, texts: List[str], is_query: bool = False
    ) -> List[List[float]]:
        """Generate 768d embeddings concurrently in batches via Cloudflare Workers AI REST API."""
        if not texts:
            return []

        if not self.account_id or not self.api_token:
            raise ValueError(
                "Cloudflare Workers AI requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN to be set."
            )

        formatted_texts = [self._format_input(t, is_query=is_query) for t in texts]
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        # Divide into batches
        batches = [
            formatted_texts[i : i + self.batch_size]
            for i in range(0, len(formatted_texts), self.batch_size)
        ]

        sem = asyncio.Semaphore(4)

        async def _fetch_single_batch(batch_idx: int, batch: List[str]) -> List[List[float]]:
            async with sem:
                client = self._get_client()
                # Cloudflare Workers AI Embedding schema supports 'input' array or 'text' array
                payload: Dict[str, Any] = {"input": batch}
                res = await client.post(url, headers=headers, json=payload)

                # Fallback to 'text' parameter if 'input' fails with 400
                if res.status_code == 400:
                    payload = {"text": batch}
                    res = await client.post(url, headers=headers, json=payload)

                if res.status_code != 200:
                    raise RuntimeError(
                        f"Cloudflare Workers AI embedding request failed with HTTP {res.status_code}: {res.text}"
                    )

                body = res.json()
                if not body.get("success", False):
                    errors = body.get("errors", [])
                    raise RuntimeError(f"Cloudflare Workers AI error: {errors}")

                result_data = body.get("result", {})
                raw_embeddings = result_data.get("data", [])

                if not raw_embeddings:
                    raise RuntimeError(
                        f"Cloudflare Workers AI returned success but empty 'data' in result: {body}"
                    )

                batch_embeddings: List[List[float]] = []
                for item in raw_embeddings:
                    if isinstance(item, list):
                        emb = [float(val) for val in item]
                    elif isinstance(item, dict) and "embedding" in item:
                        emb = [float(val) for val in item["embedding"]]
                    elif isinstance(item, dict) and "values" in item:
                        emb = [float(val) for val in item["values"]]
                    else:
                        raise ValueError(f"Unrecognized embedding item format from Cloudflare: {type(item)}")

                    if len(emb) != EMBEDDING_DIM:
                        raise ValueError(
                            f"Expected {EMBEDDING_DIM}d vector from Cloudflare '{self.model}', got {len(emb)}d"
                        )
                    batch_embeddings.append(emb)

                if len(batch_embeddings) != len(batch):
                    raise RuntimeError(
                        f"Batch {batch_idx} count mismatch: expected {len(batch)}, got {len(batch_embeddings)}"
                    )
                return batch_embeddings

        # Run batches concurrently while strictly preserving order
        batch_tasks = [_fetch_single_batch(idx, b) for idx, b in enumerate(batches)]
        batch_results = await asyncio.gather(*batch_tasks)

        all_embeddings: List[List[float]] = []
        for sublist in batch_results:
            all_embeddings.extend(sublist)

        if len(all_embeddings) != len(texts):
            raise RuntimeError(
                f"Cloudflare embedding count mismatch: expected {len(texts)}, got {len(all_embeddings)}"
            )

        return all_embeddings

    async def ensure_model_available(self) -> bool:
        """Log verification of Cloudflare Workers AI configuration."""
        if self.account_id and self.api_token:
            logger.info(
                f"Cloudflare Workers AI embedding provider initialized with model '{self.model}' (768d)."
            )
            return True
        logger.warning(
            "Cloudflare Workers AI missing credentials: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN "
            "must be configured in environment for production embeddings."
        )
        return False


class EmbeddingClientProxy:
    """Dynamic proxy that resolves and delegates to the configured embedding provider."""

    def __init__(self):
        self._provider_cache: Dict[str, BaseEmbeddingProvider] = {}

    def _resolve_provider(self) -> BaseEmbeddingProvider:
        provider_name = (settings.embedding_provider or "ollama").lower().strip()

        # Explicit Cloudflare provider
        if provider_name == "cloudflare":
            if "cloudflare" not in self._provider_cache:
                self._provider_cache["cloudflare"] = CloudflareEmbeddings()
            return self._provider_cache["cloudflare"]

        # If running in production (Render or APP_ENV=production) and Cloudflare credentials are configured,
        # automatically route to Cloudflare to guarantee independent cloud embeddings.
        is_production = settings.app_env.lower() in ("production", "prod") or bool(os.getenv("RENDER"))
        has_cf_creds = bool(settings.cloudflare_account_id and settings.cloudflare_api_token)

        if is_production and has_cf_creds:
            if "cloudflare" not in self._provider_cache:
                self._provider_cache["cloudflare"] = CloudflareEmbeddings()
            return self._provider_cache["cloudflare"]

        # Default is Ollama (Local Development: nomic-embed-text, 768d) - 100% strictly preserved
        if "ollama" not in self._provider_cache:
            self._provider_cache["ollama"] = OllamaEmbeddings()
        return self._provider_cache["ollama"]

    @property
    def model_name(self) -> str:
        return self._resolve_provider().model_name

    @property
    def dimension(self) -> int:
        return self._resolve_provider().dimension

    async def get_embedding(self, text: str, is_query: bool = True) -> List[float]:
        return await self._resolve_provider().get_embedding(text, is_query=is_query)

    async def get_embeddings_batch(
        self, texts: List[str], is_query: bool = False
    ) -> List[List[float]]:
        return await self._resolve_provider().get_embeddings_batch(texts, is_query=is_query)

    async def ensure_model_available(self) -> bool:
        return await self._resolve_provider().ensure_model_available()


# Module-level exported singleton & legacy constants for backwards compatibility
embeddings_client = EmbeddingClientProxy()
EMBEDDING_MODEL = settings.embedding_model or "nomic-embed-text"
