"""TARK Studio — Embeddings Client for Ollama nomic-embed-text (768d).

Generates dense semantic vector embeddings for document chunks and user queries.
Uses identical model and parameters across ingestion and retrieval.
"""

import logging
from typing import List, Optional
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
DEFAULT_BATCH_SIZE = 32


class OllamaEmbeddings:
    """Asynchronous client for Ollama embedding generation."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a single query or text passage."""
        results = await self.get_embeddings_batch([text])
        if not results:
            raise RuntimeError(f"Ollama failed to return an embedding for prompt.")
        return results[0]

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batches using Ollama /api/embed."""
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
                        raise RuntimeError(f"Ollama embedding request failed with HTTP {res.status_code}: {res.text}")
                    emb = res.json().get("embedding")
                    if not emb or len(emb) != EMBEDDING_DIM:
                        raise ValueError(f"Expected {EMBEDDING_DIM}d embedding from {self.model}, got {len(emb) if emb else 0}d")
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


# Default singleton instance
embeddings_client = OllamaEmbeddings()
