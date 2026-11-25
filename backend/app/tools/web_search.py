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
_MAX_SNIPPET_CHARS = 1000
_MIN_CACHEABLE_SCORE = 0.35
_ESCALATION_SCORE_THRESHOLD = 0.40
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "what", "who", "when", "where",
    "which", "how", "many", "much", "till", "today", "latest", "current", "currently",
    "tell", "search", "find", "give", "show", "about", "are", "was", "were", "has", "have",
})
_SOCIAL_NOISE_DOMAINS = (
    "facebook.com",
    "linkedin.com",
    "instagram.com",
    "pinterest.com",
    "tiktok.com",
    "tracxn.com",
    "threads.com",
    "threads.net",
)
_JUNK_BOILERPLATE_MARKERS = (
    "javascript is disabled in your browser",
    "enable javascript to view",
    "log in or sign up to view",
    "create an account to continue",
)
_active_web_cache_tasks: set[asyncio.Task] = set()


def _should_exclude_social(query_text: str) -> bool:
    """Exclude social profile/login-wall directories unless explicitly requested."""
    lowered = (query_text or "").lower()
    return not any(dom.split(".")[0] in lowered for dom in _SOCIAL_NOISE_DOMAINS)


def _is_junk_snippet(content: str) -> bool:
    lowered = (content or "").lower()
    return any(marker in lowered for marker in _JUNK_BOILERPLATE_MARKERS)


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
    def _build_cache_key(
        query: str,
        topic: str,
        search_depth: str = "basic",
        exclude_social: bool = True,
    ) -> str:
        norm = " ".join((query or "").strip().lower().split())
        q_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        ex_tag = "ex" if exclude_social else "all"
        return f"tark:web:v2:{topic}:{search_depth}:{ex_tag}:{q_hash}"

    async def _get_cached_results(
        self,
        query: str,
        topic: str,
        search_depth: str = "basic",
        exclude_social: bool = True,
    ) -> Optional[List[WebSearchResult]]:
        client = await redis_cache.get_client()
        if client is None:
            return None
        try:
            raw = await client.get(
                self._build_cache_key(query, topic, search_depth=search_depth, exclude_social=exclude_social)
            )
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
        self,
        query: str,
        topic: str,
        results: List[WebSearchResult],
        search_depth: str = "basic",
        exclude_social: bool = True,
    ) -> None:
        if not results or max((r.score for r in results), default=0.0) < _MIN_CACHEABLE_SCORE:
            return
        client = await redis_cache.get_client()
        if client is None:
            return
        try:
            payload = json.dumps([asdict(r) for r in results], ensure_ascii=False).encode("utf-8")
            ttl = max(60, int(settings.web_search_cache_ttl_seconds or 900))
            key = self._build_cache_key(
                query, topic, search_depth=search_depth, exclude_social=exclude_social
            )
            await client.setex(key, ttl, payload)
        except Exception as exc:
            logger.debug(f"[WebSearch] Redis cache write skipped: {exc}")

    async def _search_single_tavily(
        self,
        query: str,
        topic: str = "general",
        max_results: int = 5,
        search_depth: str = "basic",
        exclude_social: bool = True,
    ) -> List[WebSearchResult]:
        """Execute a single Tavily REST search with 15m Redis caching."""
        cached = await self._get_cached_results(
            query, topic, search_depth=search_depth, exclude_social=exclude_social
        )
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
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max(1, min(10, int(max_results))),
            "include_answer": False,
            "include_raw_content": False,
        }
        if exclude_social:
            payload["exclude_domains"] = list(_SOCIAL_NOISE_DOMAINS)

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
                if not content or _is_junk_snippet(content):
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
                _schedule_web_cache_write(
                    self._set_cached_results(
                        query,
                        topic,
                        parsed,
                        search_depth=search_depth,
                        exclude_social=exclude_social,
                    )
                )
            return parsed
        except Exception as exc:
            logger.warning(f"[WebSearch] Tavily request failed (fail-open): {exc}")
            return []

    @staticmethod
    def _merge_results(lists: List[List[WebSearchResult]]) -> List[WebSearchResult]:
        by_url: Dict[str, WebSearchResult] = {}
        for lst in lists:
            for item in lst:
                norm_url = item.url.rstrip("/")
                if norm_url not in by_url or item.score > by_url[norm_url].score:
                    by_url[norm_url] = item
        return sorted(by_url.values(), key=lambda x: x.score, reverse=True)

    @staticmethod
    def _rank_results(
        results: List[WebSearchResult], user_query: str, max_k: int
    ) -> List[WebSearchResult]:
        """Fast (<1ms) generalized web ranker combining Tavily score, term coverage, and domain diversity."""
        if not results:
            return []
        tokens = [
            t for t in re.findall(r"[a-z0-9]{3,}", (user_query or "").lower())
            if t not in _STOPWORDS
        ]

        def _rank_key(r: WebSearchResult) -> float:
            score = float(r.score)
            if tokens:
                title_l = (r.title or "").lower()
                haystack = f"{title_l} {(r.content or '').lower()}"
                kw_ratio = sum(1 for tok in tokens if tok in haystack) / len(tokens)
                title_ratio = sum(1 for tok in tokens if tok in title_l) / len(tokens)
                score += 0.14 * kw_ratio + 0.06 * title_ratio
            if len(r.content or "") < 120:
                score -= 0.15
            dom = (r.domain or "").lower()
            if dom.endswith((".gov", ".edu")) or "wikipedia.org" in dom or dom.startswith("docs."):
                score += 0.04
            return score

        ranked = sorted(results, key=_rank_key, reverse=True)
        # Keep source diversity: max 2 snippets per domain before filling remaining slots
        selected: List[WebSearchResult] = []
        overflow: List[WebSearchResult] = []
        domain_counts: Dict[str, int] = {}
        for item in ranked:
            dom = (item.domain or item.url).lower()
            cnt = domain_counts.get(dom, 0)
            if cnt < 2:
                selected.append(item)
                domain_counts[dom] = cnt + 1
            else:
                overflow.append(item)
            if len(selected) >= max_k:
                return selected[:max_k]
        return (selected + overflow)[:max_k]

    async def search(
        self,
        queries: List[str],
        user_query: str,
        topic: str = "general",
        max_final_results: Optional[int] = None,
        use_reranker: bool = True,
    ) -> tuple[List[WebSearchResult], float]:
        """Run up to 2 search queries in parallel, self-correct weak hits, deduplicate, and rank."""
        t0 = time.perf_counter()
        clean_queries = [q.strip() for q in queries if q and q.strip()][:2]
        if not clean_queries:
            return [], 0.0

        max_k = max_final_results or int(settings.web_search_max_results or 5)
        fetch_k = max_k if len(clean_queries) > 1 else min(8, max_k + 2)
        exclude_social = _should_exclude_social(user_query)
        batch_lists = await asyncio.gather(
            *(
                self._search_single_tavily(
                    q,
                    topic=topic,
                    max_results=fetch_k,
                    search_depth="basic",
                    exclude_social=exclude_social,
                )
                for q in clean_queries
            )
        )

        merged = self._merge_results(list(batch_lists))
        max_score = max((r.score for r in merged), default=0.0)

        # Self-corrective escalation (CRAG-lite): when Hop 1 returns sparse or weak sources
        # (common when `news` index misses reference/stats pages), retry once in general/advanced mode.
        if len(merged) < 2 or max_score < _ESCALATION_SCORE_THRESHOLD:
            logger.info(
                f"[WebSearch] Escalating weak Hop-1 results (count={len(merged)}, max_score={max_score:.3f}, "
                f"topic='{topic}') -> topic='general', search_depth='advanced'"
            )
            retry_results = await self._search_single_tavily(
                clean_queries[0],
                topic="general",
                max_results=max_k,
                search_depth="advanced",
                exclude_social=exclude_social,
            )
            if retry_results:
                merged = self._merge_results([merged, retry_results])

        final_results = self._rank_results(merged, user_query, max_k)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return final_results, latency_ms


web_search_client = WebSearchClient()
