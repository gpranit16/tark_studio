"""TARK Studio — Agentic Web Search Tool (Tavily + Redis 15m Cache + ONNX Relevance Filter).

Security & Reliability guarantees (AGENTS.md Sec 5):
1. Treats all fetched web pages/snippets as untrusted external content; strips HTML/scripts
   and neutralizes prompt-delimiter injection tokens.
2. Enforces strict HTTP timeouts and bounded snippet lengths.
3. Caches raw web search results in Redis for short TTL (15m default) with fail-open fallback.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from app.config import settings
from app.rag.cache import redis_cache

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_MAX_SNIPPET_CHARS = 1200
_active_web_cache_tasks: set[asyncio.Task] = set()


def _schedule_web_cache_write(coro) -> None:
    task = asyncio.create_task(coro)
    _active_web_cache_tasks.add(task)
    task.add_done_callback(_active_web_cache_tasks.discard)


@dataclass
class WebSearchResult:
    """Sanitized external web search snippet with provenance metadata."""

    title: str
    url: str
    domain: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None
    rerank_score: Optional[float] = None


def sanitize_web_text(text: str, max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    """Strip HTML/scripts, control chars, and delimiter breakout tags from untrusted web text."""
    if not text:
        return ""
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Neutralize boundary tags that could spoof trusted system delimiters
    cleaned = re.sub(r"</?untrusted_web_context[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_chars]


def _extract_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


class WebSearchClient:
    """Pooled async Tavily search client with Redis 15m cache and fail-open behavior."""

    def __init__(self) -> None:
        self._http_client: Optional[httpx.AsyncClient] = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            timeout = float(settings.web_search_timeout_sec or 5.0)
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=2.5),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0),
            )
        return self._http_client

    async def close(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    @staticmethod
    def _build_cache_key(query: str, topic: str) -> str:
        norm = " ".join((query or "").strip().lower().split())
        q_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        return f"tark:web:{topic}:{q_hash}"

    async def _get_cached_results(self, query: str, topic: str) -> Optional[List[WebSearchResult]]:
        client = await redis_cache.get_client()
        if client is None:
            return None
        try:
            raw = await client.get(self._build_cache_key(query, topic))
            if not raw:
                return None
            items = json.loads(raw.decode("utf-8"))
            if not isinstance(items, list):
                return None
            return [WebSearchResult(**item) for item in items if isinstance(item, dict)]
        except Exception as exc:
            logger.debug(f"[WebSearch] Redis cache read skipped: {exc}")
            return None

    async def _set_cached_results(
        self, query: str, topic: str, results: List[WebSearchResult]
    ) -> None:
        if not results:
            return
        client = await redis_cache.get_client()
        if client is None:
            return
        try:
            payload = json.dumps([asdict(r) for r in results], ensure_ascii=False).encode("utf-8")
            ttl = max(60, int(settings.web_search_cache_ttl_seconds or 900))
            await client.setex(self._build_cache_key(query, topic), ttl, payload)
        except Exception as exc:
            logger.debug(f"[WebSearch] Redis cache write skipped: {exc}")

    async def _search_single_tavily(
        self, query: str, topic: str = "general", max_results: int = 5
    ) -> List[WebSearchResult]:
        """Execute a single Tavily REST search with 15m Redis caching."""
        cached = await self._get_cached_results(query, topic)
        if cached is not None:
            logger.info(f"[WebSearch] Redis cache HIT for query='{query[:60]}' ({len(cached)} results)")
            return cached

        api_key = (settings.tavily_api_key or "").strip()
        if not api_key:
            logger.warning("[WebSearch] TAVILY_API_KEY is not configured; returning empty web results.")
            return []

        payload: Dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "topic": topic,
            "max_results": max(1, min(10, int(max_results))),
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            client = self._get_http_client()
            res = await client.post(TAVILY_SEARCH_URL, json=payload)
            if res.status_code != 200:
                logger.warning(f"[WebSearch] Tavily returned HTTP {res.status_code}: {res.text[:200]}")
                return []

            body = res.json()
            raw_results = body.get("results") or []
            parsed: List[WebSearchResult] = []
            for item in raw_results:
                url = str(item.get("url") or "").strip()
                if not url.startswith(("http://", "https://")):
                    continue
                content = sanitize_web_text(str(item.get("content") or ""))
                title = sanitize_web_text(str(item.get("title") or url), max_chars=240)
                if not content:
                    continue
                parsed.append(
                    WebSearchResult(
                        title=title,
                        url=url,
                        domain=_extract_domain(url),
                        content=content,
                        score=float(item.get("score") or 0.0),
                        published_date=item.get("published_date"),
                    )
                )

            if parsed:
                _schedule_web_cache_write(self._set_cached_results(query, topic, parsed))
            return parsed
        except Exception as exc:
            logger.warning(f"[WebSearch] Tavily request failed (fail-open): {exc}")
            return []

    async def search(
        self,
        queries: List[str],
        user_query: str,
        topic: str = "general",
        max_final_results: Optional[int] = None,
        use_reranker: bool = True,
    ) -> tuple[List[WebSearchResult], float]:
        """Run up to 2 search queries in parallel, deduplicate URLs, and rank by relevance."""
        t0 = time.perf_counter()
        clean_queries = [q.strip() for q in queries if q and q.strip()][:2]
        if not clean_queries:
            return [], 0.0

        max_k = max_final_results or int(settings.web_search_max_results or 5)
        batch_lists = await asyncio.gather(
            *(self._search_single_tavily(q, topic=topic, max_results=max_k) for q in clean_queries)
        )

        # Deduplicate by normalized URL preserving highest Tavily score
        by_url: Dict[str, WebSearchResult] = {}
        for lst in batch_lists:
            for item in lst:
                norm_url = item.url.rstrip("/")
                if norm_url not in by_url or item.score > by_url[norm_url].score:
                    by_url[norm_url] = item

        merged = sorted(by_url.values(), key=lambda x: x.score, reverse=True)

        # Use ONNX Cross-Encoder only when already warm in RAM and >max_k multi-query results need pruning
        if use_reranker and len(merged) > max_k:
            try:
                from app.rag.reranker import _RERANKER_CACHE, get_reranker

                if "onnx" in _RERANKER_CACHE:
                    reranker = get_reranker("onnx")
                    candidate_dicts = [{"content": f"{r.title}. {r.content}", "idx": i} for i, r in enumerate(merged)]
                    reranked = await asyncio.to_thread(reranker.rerank, user_query, candidate_dicts, max_k)
                    ordered: List[WebSearchResult] = []
                    for rc in reranked:
                        orig = merged[int(rc["idx"])]
                        orig.rerank_score = rc.get("rerank_score")
                        ordered.append(orig)
                    merged = ordered
            except Exception as exc:
                logger.debug(f"[WebSearch] ONNX snippet filter skipped: {exc}")

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return merged[:max_k], latency_ms


web_search_client = WebSearchClient()
