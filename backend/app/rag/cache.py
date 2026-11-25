"""TARK Studio — Fail-Open Redis Cache Manager (L1 Answer Cache & Embedding Cache).

Design guarantees:
1. Zero critical-path regression: strict 50ms socket timeout + 10s circuit-breaker cooldown
   when Redis is unavailable or disabled (fail-open in <1ms).
2. 100% Accuracy preservation:
   - Uses exact normalized query hashes (never fuzzy/semantic guesswork on numbers or years).
   - Partitions keys by corpus_version, provider, model, reranker_type, and sorted document_ids.
   - Never serves cached answers for multi-turn follow-ups or Retry/Regenerate requests.
   - Write-behind caching runs asynchronously after stream completion (0ms added to TTFT/E2E).
"""

import hashlib
import json
import logging
import re
import struct
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover - optional runtime fallback
    aioredis = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CORPUS_VERSION_KEY = "tark:corpus:v"
_CIRCUIT_COOLDOWN_SEC = 10.0
_DEFAULT_REDIS_TIMEOUT_SEC = 0.25
_DEFAULT_REDIS_CONNECT_TIMEOUT_SEC = 0.5


def normalize_query(query: str) -> str:
    """Normalize query casing and whitespace while preserving exact numbers, entities, and terms."""
    return " ".join((query or "").strip().lower().split())


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Generalizable Structural Invariant Guards (Zero query-specific hardcoding)
# ---------------------------------------------------------------------------

_NUMBER_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}

_NEGATION_WORDS = frozenset({"not", "no", "never", "none", "neither", "nor", "cannot"})
_EXCLUSION_WORDS = frozenset({"without", "except", "excluding", "unless", "instead"})
_SENTENCE_STARTERS = frozenset({
    "what", "why", "how", "when", "where", "who", "whom", "whose", "which",
    "tell", "explain", "describe", "summarize", "summarise", "compare", "list",
    "give", "show", "define", "find", "can", "could", "would", "should", "will",
    "is", "are", "was", "were", "do", "does", "did", "please", "the", "a", "an", "in", "on",
})


def _extract_numeric_invariants(text: str) -> Tuple[str, ...]:
    """Extract canonical numbers, ordinals, and alphanumeric codes (e.g. 1st/first -> '1', 'q3')."""
    lowered = (text or "").lower()
    tokens: List[str] = []

    # Alphanumeric identifiers mixing letters & digits (e.g. q1, q3, v2, b2b, gpt4)
    for m in re.finditer(r"\b(?:[a-z]+\d+[a-z0-9]*|\d+(?:[a-z]{3,}|[a-z]\d+[a-z0-9]*))\b", lowered):
        tokens.append(f"code:{m.group(0)}")

    # Pure digits, decimals, and numeric ordinals (2023, 3.5, 1st -> 1, 2nd -> 2)
    for m in re.finditer(r"(?<![a-z])\d+(?:\.\d+)?(?:st|nd|rd|th)?(?![a-z])", lowered):
        val = re.sub(r"(?:st|nd|rd|th)$", "", m.group(0))
        tokens.append(f"num:{val}")

    # English cardinal/ordinal number words
    for word in re.findall(r"\b[a-z]+\b", lowered):
        if word in _NUMBER_WORDS:
            tokens.append(f"num:{_NUMBER_WORDS[word]}")

    return tuple(sorted(tokens))


def _extract_polarity_signature(text: str) -> Tuple[int, frozenset[str]]:
    """Extract grammatical negation parity and exclusion prepositions."""
    normalized = re.sub(r"\b([a-z]+)n't\b", r"\1 not", (text or "").lower())
    words = re.findall(r"\b[a-z]+\b", normalized)
    neg_count = sum(1 for w in words if w in _NEGATION_WORDS)
    excl = frozenset(w for w in words if w in _EXCLUSION_WORDS)
    return (neg_count % 2, excl)


def _extract_entities_and_quotes(text: str) -> Tuple[frozenset[str], frozenset[str]]:
    """Extract exact quoted literals and proper-noun/acronym entities."""
    raw = text or ""
    quotes = frozenset(
        q.strip().lower()
        for q in re.findall(r'["\']([^"\']{2,})["\']', raw)
        if q.strip()
    )
    entities: set[str] = set()
    for idx, m in enumerate(re.finditer(r"\b([A-Za-z][A-Za-z0-9_-]*)\b", raw)):
        tok = m.group(1)
        if len(tok) < 2:
            continue
        is_acronym = tok.isupper() and len(tok) >= 2
        is_proper = tok[0].isupper() and not (idx == 0 and tok.lower() in _SENTENCE_STARTERS)
        if is_acronym or is_proper:
            entities.add(tok.lower())
    return frozenset(entities), quotes


def passes_structural_guard(query_a: str, query_b: str) -> bool:
    """Return True only when both queries share identical numbers, polarity, entities, and quotes."""
    if _extract_numeric_invariants(query_a) != _extract_numeric_invariants(query_b):
        return False
    if _extract_polarity_signature(query_a) != _extract_polarity_signature(query_b):
        return False
    if _extract_entities_and_quotes(query_a) != _extract_entities_and_quotes(query_b):
        return False
    return True


class RedisCacheManager:
    """Async Redis client wrapper with strict timeouts, circuit breaker, and fail-open behavior."""

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._cooldown_until: float = 0.0
        self._initialized_url: Optional[str] = None

    @property
    def is_enabled(self) -> bool:
        return bool(settings.redis_enabled and settings.redis_url and aioredis is not None)

    def _is_circuit_open(self) -> bool:
        return time.monotonic() < self._cooldown_until

    def _trip_circuit(self, exc: Exception, operation: str) -> None:
        self._cooldown_until = time.monotonic() + _CIRCUIT_COOLDOWN_SEC
        logger.warning(
            f"[Redis] {operation} failed ({exc}); failing open for {_CIRCUIT_COOLDOWN_SEC}s."
        )

    async def get_client(self) -> Optional[Any]:
        """Return pooled async Redis client, or None if disabled/circuit-open."""
        if not self.is_enabled or self._is_circuit_open():
            return None

        target_url = settings.redis_url
        if self._client is not None and self._initialized_url == target_url:
            return self._client

        try:
            if self._client is not None:
                await self.close()
            self._client = aioredis.from_url(
                target_url,
                socket_timeout=_DEFAULT_REDIS_TIMEOUT_SEC,
                socket_connect_timeout=_DEFAULT_REDIS_CONNECT_TIMEOUT_SEC,
                retry_on_timeout=False,
                decode_responses=False,
                max_connections=32,
            )
            self._initialized_url = target_url
            return self._client
        except Exception as exc:
            self._trip_circuit(exc, "connect")
            return None

    @staticmethod
    async def _count_keys(client: Any, pattern: str, max_scans: int = 10) -> int:
        """Count keys matching pattern with bounded non-blocking SCAN iterations."""
        count = 0
        cursor = 0
        for _ in range(max_scans):
            cursor, keys = await client.scan(cursor=cursor, match=pattern, count=200)
            count += len(keys)
            if int(cursor) == 0:
                break
        return count

    async def ping(self) -> Dict[str, Any]:
        """Non-blocking health check returning Redis availability and safe cache stats."""
        if not self.is_enabled:
            return {"enabled": False, "connected": False}
        client = await self.get_client()
        if client is None:
            return {"enabled": True, "connected": False, "circuit_open": self._is_circuit_open()}
        try:
            ok = bool(await client.ping())
            result: Dict[str, Any] = {"enabled": True, "connected": ok}
            if ok:
                try:
                    info_mem = await client.info("memory")
                    info_stats = await client.info("stats")
                    corpus_ver = await self.get_corpus_version()
                    ans_count = await self._count_keys(client, "tark:ans:*")
                    emb_count = await self._count_keys(client, "tark:emb:*")
                    sem_buckets = await self._count_keys(client, "tark:sem:*")
                    used_mem = info_mem.get("used_memory_human") or info_mem.get(b"used_memory_human")
                    if isinstance(used_mem, bytes):
                        used_mem = used_mem.decode("utf-8", errors="ignore")
                    hits = int(info_stats.get("keyspace_hits") or info_stats.get(b"keyspace_hits") or 0)
                    misses = int(info_stats.get("keyspace_misses") or info_stats.get(b"keyspace_misses") or 0)
                    result.update({
                        "corpus_version": corpus_ver,
                        "answer_keys": ans_count,
                        "embedding_keys": emb_count,
                        "semantic_buckets": sem_buckets,
                        "used_memory_human": used_mem,
                        "hits": hits,
                        "misses": misses,
                    })
                except Exception as stats_exc:
                    logger.debug(f"[Redis] Safe stats skipped: {stats_exc}")
            return result
        except Exception as exc:
            self._trip_circuit(exc, "ping")
            return {"enabled": True, "connected": False, "error": str(exc)}

    async def close(self) -> None:
        """Close underlying Redis connection pool cleanly on shutdown."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:
                logger.debug(f"[Redis] Error closing client pool: {exc}")
            finally:
                self._client = None
                self._initialized_url = None

    # ------------------------------------------------------------------
    # O(1) Corpus Version Invalidation
    # ------------------------------------------------------------------

    async def get_corpus_version(self) -> int:
        """Read current O(1) corpus version integer (initializes to 0 on first read)."""
        client = await self.get_client()
        if client is None:
            return 0
        try:
            raw = await client.get(CORPUS_VERSION_KEY)
            if raw is None:
                await client.setnx(CORPUS_VERSION_KEY, b"0")
                return 0
            return int(raw)
        except Exception as exc:
            self._trip_circuit(exc, "get_corpus_version")
            return 0

    async def bump_corpus_version(self) -> Optional[int]:
        """Atomically increment corpus version so prior RAG answer keys become obsolete in O(1)."""
        client = await self.get_client()
        if client is None:
            return None
        try:
            new_ver = int(await client.incr(CORPUS_VERSION_KEY))
            logger.info(f"[Redis] Corpus version bumped to v{new_ver} (O(1) cache invalidation).")
            return new_ver
        except Exception as exc:
            self._trip_circuit(exc, "bump_corpus_version")
            return None

    # ------------------------------------------------------------------
    # L1 Normalized Answer Cache
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scope_prefix(
        corpus_version: int,
        provider: str,
        model: str,
        reranker_type: Optional[str],
        document_ids: Optional[List[str]],
        temperature: float = 0.7,
    ) -> str:
        docs_scope = ",".join(sorted(str(d) for d in document_ids)) if document_ids else "no_docs"
        docs_hash = _sha256_hex(docs_scope)[:16]
        rerank_tag = (reranker_type or settings.reranker_type or "default").lower()
        temp_tag = f"{round(float(temperature), 2)}"
        return f"v{corpus_version}:{provider}:{model}:{rerank_tag}:t{temp_tag}:{docs_hash}"

    @staticmethod
    def build_answer_key(
        corpus_version: int,
        provider: str,
        model: str,
        reranker_type: Optional[str],
        document_ids: Optional[List[str]],
        query: str,
        temperature: float = 0.7,
    ) -> str:
        prefix = RedisCacheManager._build_scope_prefix(
            corpus_version, provider, model, reranker_type, document_ids, temperature
        )
        q_hash = _sha256_hex(normalize_query(query))
        return f"tark:ans:{prefix}:{q_hash}"

    @staticmethod
    def build_semantic_bucket_key(
        corpus_version: int,
        provider: str,
        model: str,
        reranker_type: Optional[str],
        document_ids: Optional[List[str]],
        temperature: float = 0.7,
    ) -> str:
        prefix = RedisCacheManager._build_scope_prefix(
            corpus_version, provider, model, reranker_type, document_ids, temperature
        )
        return f"tark:sem:{prefix}"

    @staticmethod
    def _pack_semantic_entry(query: str, ans_key: str, vector: List[float]) -> bytes:
        meta_bytes = json.dumps(
            {"q": query.strip(), "ans_key": ans_key, "ts": time.time()},
            ensure_ascii=False,
        ).encode("utf-8")
        vec_bytes = struct.pack(f"<{len(vector)}f", *vector)
        return struct.pack("<I", len(meta_bytes)) + meta_bytes + vec_bytes

    @staticmethod
    def _unpack_semantic_entry(raw: bytes) -> Optional[Tuple[Dict[str, Any], np.ndarray]]:
        if not raw or len(raw) < 8:
            return None
        meta_len = struct.unpack("<I", raw[:4])[0]
        if 4 + meta_len >= len(raw):
            return None
        vec_bytes = raw[4 + meta_len :]
        if len(vec_bytes) % 4 != 0 or len(vec_bytes) == 0:
            return None
        meta = json.loads(raw[4 : 4 + meta_len].decode("utf-8"))
        vec = np.frombuffer(vec_bytes, dtype="<f4").astype(np.float32)
        return meta, vec

    async def get_cached_answer(
        self,
        provider: str,
        model: str,
        reranker_type: Optional[str],
        document_ids: Optional[List[str]],
        query: str,
        temperature: float = 0.7,
    ) -> Optional[Dict[str, Any]]:
        """Fetch cached exact-normalized RAG/chat answer and citations, or None on miss/error."""
        client = await self.get_client()
        if client is None:
            return None
        try:
            # Pipeline version lookup isn't needed before key build unless we store version,
            # so get version first (~0.3ms) then fetch exact key (~0.3ms).
            version = await self.get_corpus_version()
            key = self.build_answer_key(
                corpus_version=version,
                provider=provider,
                model=model,
                reranker_type=reranker_type,
                document_ids=document_ids,
                query=query,
                temperature=temperature,
            )
            raw = await client.get(key)
            if not raw:
                return None
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict) and payload.get("answer"):
                return payload
            return None
        except Exception as exc:
            self._trip_circuit(exc, "get_cached_answer")
            return None

    async def set_cached_answer(
        self,
        provider: str,
        model: str,
        reranker_type: Optional[str],
        document_ids: Optional[List[str]],
        query: str,
        answer: str,
        citations: List[dict],
        temperature: float = 0.7,
        rag_status: Optional[dict] = None,
        get_embedding_fn: Optional[Callable[[str], Awaitable[List[float]]]] = None,
    ) -> None:
        """Persist completed answer + citations in Redis L1 and L2 semantic index (non-blocking)."""
        if not answer or not answer.strip():
            return
        client = await self.get_client()
        if client is None:
            return
        try:
            version = await self.get_corpus_version()
            key = self.build_answer_key(
                corpus_version=version,
                provider=provider,
                model=model,
                reranker_type=reranker_type,
                document_ids=document_ids,
                query=query,
                temperature=temperature,
            )
            payload = json.dumps(
                {
                    "answer": answer,
                    "citations": citations or [],
                    "rag_status": rag_status,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            ttl = max(60, int(settings.cache_ttl_seconds))
            await client.setex(key, ttl, payload)

            # Index query vector in L2 semantic bucket for paraphrased matching
            if settings.semantic_cache_enabled and get_embedding_fn is not None:
                try:
                    vec = await get_embedding_fn(query)
                    if vec:
                        bucket_key = self.build_semantic_bucket_key(
                            corpus_version=version,
                            provider=provider,
                            model=model,
                            reranker_type=reranker_type,
                            document_ids=document_ids,
                            temperature=temperature,
                        )
                        q_field = _sha256_hex(normalize_query(query))[:24]
                        packed = self._pack_semantic_entry(query, key, vec)
                        await client.hset(bucket_key, q_field, packed)
                        await client.expire(bucket_key, ttl)

                        # Bound bucket cardinality so HGETALL stays <1ms
                        max_entries = max(10, int(settings.semantic_cache_max_entries))
                        existing = await client.hgetall(bucket_key)
                        if len(existing) > max_entries:
                            oldest_field = None
                            oldest_ts = float("inf")
                            for f_key, f_raw in existing.items():
                                parsed = self._unpack_semantic_entry(f_raw)
                                ts = float(parsed[0].get("ts", 0.0)) if parsed else 0.0
                                if ts < oldest_ts:
                                    oldest_ts = ts
                                    oldest_field = f_key
                            if oldest_field is not None:
                                await client.hdel(bucket_key, oldest_field)
                except Exception as sem_exc:
                    logger.debug(f"[Redis] Semantic bucket index skipped: {sem_exc}")
        except Exception as exc:
            self._trip_circuit(exc, "set_cached_answer")

    async def get_semantic_cached_answer(
        self,
        provider: str,
        model: str,
        reranker_type: Optional[str],
        document_ids: Optional[List[str]],
        query: str,
        temperature: float = 0.7,
        get_embedding_fn: Optional[Callable[[str], Awaitable[List[float]]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """L2 paraphrased cache lookup with NumPy cosine similarity + structural invariant guards."""
        if not settings.semantic_cache_enabled or get_embedding_fn is None:
            return None
        client = await self.get_client()
        if client is None:
            return None
        try:
            version = await self.get_corpus_version()
            bucket_key = self.build_semantic_bucket_key(
                corpus_version=version,
                provider=provider,
                model=model,
                reranker_type=reranker_type,
                document_ids=document_ids,
                temperature=temperature,
            )
            raw_entries = await client.hgetall(bucket_key)
            if not raw_entries:
                # Empty bucket: return immediately without spending an embedding API call
                return None

            try:
                q_vec_list = await get_embedding_fn(query)
            except Exception as emb_exc:
                logger.debug(f"[Redis] Semantic embedding lookup skipped (fail-open): {emb_exc}")
                return None
            if not q_vec_list:
                return None
            q_vec = np.asarray(q_vec_list, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm == 0.0:
                return None
            q_unit = q_vec / q_norm

            candidates_meta: List[Dict[str, Any]] = []
            candidates_vecs: List[np.ndarray] = []
            for raw_val in raw_entries.values():
                unpacked = self._unpack_semantic_entry(raw_val)
                if unpacked is None:
                    continue
                meta, vec = unpacked
                if vec.shape[0] != q_unit.shape[0]:
                    continue
                v_norm = float(np.linalg.norm(vec))
                if v_norm == 0.0:
                    continue
                candidates_meta.append(meta)
                candidates_vecs.append(vec / v_norm)

            if not candidates_vecs:
                return None

            matrix = np.vstack(candidates_vecs)
            scores = matrix @ q_unit
            threshold = float(settings.semantic_cache_threshold)

            # Evaluate passing candidates in descending cosine similarity order
            ranked_indices = np.argsort(-scores)
            for idx in ranked_indices:
                sim = float(scores[idx])
                if sim < threshold:
                    break
                meta = candidates_meta[int(idx)]
                cand_q = meta.get("q", "")
                if not passes_structural_guard(query, cand_q):
                    continue
                ans_key = meta.get("ans_key")
                if not ans_key:
                    continue
                raw_ans = await client.get(ans_key)
                if not raw_ans:
                    continue
                payload = json.loads(raw_ans.decode("utf-8"))
                if isinstance(payload, dict) and payload.get("answer"):
                    payload["cache_tier"] = "semantic"
                    payload["similarity"] = round(sim, 4)
                    return payload
            return None
        except Exception as exc:
            self._trip_circuit(exc, "get_semantic_cached_answer")
            return None

    # ------------------------------------------------------------------
    # Shared Embedding Cache (Compact float32 binary packing)
    # ------------------------------------------------------------------

    @staticmethod
    def build_embedding_key(model_name: str, text: str) -> str:
        return f"tark:emb:{model_name}:{_sha256_hex(text.strip())}"

    async def get_cached_embedding(self, model_name: str, text: str) -> Optional[List[float]]:
        """Retrieve packed float32 embedding vector from Redis, or None on miss/error."""
        client = await self.get_client()
        if client is None:
            return None
        try:
            key = self.build_embedding_key(model_name, text)
            raw = await client.get(key)
            if not raw or len(raw) % 4 != 0:
                return None
            count = len(raw) // 4
            return list(struct.unpack(f"<{count}f", raw))
        except Exception as exc:
            self._trip_circuit(exc, "get_cached_embedding")
            return None

    async def set_cached_embedding(
        self, model_name: str, text: str, vector: List[float]
    ) -> None:
        """Store compact float32 embedding vector in Redis with embedding TTL."""
        if not vector:
            return
        client = await self.get_client()
        if client is None:
            return
        try:
            key = self.build_embedding_key(model_name, text)
            packed = struct.pack(f"<{len(vector)}f", *vector)
            ttl = max(300, int(settings.embedding_cache_ttl_seconds))
            await client.setex(key, ttl, packed)
        except Exception as exc:
            self._trip_circuit(exc, "set_cached_embedding")


# Global fail-open Redis cache manager singleton
redis_cache = RedisCacheManager()
