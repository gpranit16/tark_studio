"""Unit and integration tests for TARK Studio fail-open Redis cache."""

import asyncio
import json
import uuid
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import settings
from app.rag.cache import (
    RedisCacheManager,
    normalize_query,
    passes_structural_guard,
    redis_cache,
    CORPUS_VERSION_KEY,
)


class InMemoryFakeRedis:
    """Lightweight async Redis double supporting get, setex, setnx, incr, hash, and scan."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.ttls: dict[str, int] = {}

    async def ping(self) -> bool:
        return True

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: bytes):
        self.store[key] = value
        self.ttls[key] = ttl
        return True

    async def setnx(self, key: str, value: bytes):
        if key not in self.store:
            self.store[key] = value
            return True
        return False

    async def incr(self, key: str) -> int:
        cur = int(self.store.get(key, b"0"))
        nxt = cur + 1
        self.store[key] = str(nxt).encode("utf-8")
        return nxt

    async def scan(self, cursor: int = 0, match: str = "*", count: int = 200):
        prefix = match.rstrip("*")
        matched = [k.encode("utf-8") for k in list(self.store.keys()) + list(self.hashes.keys()) if k.startswith(prefix)]
        return 0, matched

    async def hset(self, key: str, field: str, value: bytes):
        bucket = self.hashes.setdefault(key, {})
        f_b = field.encode("utf-8") if isinstance(field, str) else field
        bucket[f_b] = value
        return 1

    async def hgetall(self, key: str):
        return dict(self.hashes.get(key, {}))

    async def hdel(self, key: str, *fields):
        bucket = self.hashes.get(key, {})
        removed = 0
        for f in fields:
            f_b = f.encode("utf-8") if isinstance(f, str) else f
            if f_b in bucket:
                del bucket[f_b]
                removed += 1
        return removed

    async def expire(self, key: str, ttl: int):
        self.ttls[key] = ttl
        return True

    async def info(self, section: str = ""):
        if section == "memory":
            return {"used_memory_human": "1.00M"}
        if section == "stats":
            return {"keyspace_hits": 3, "keyspace_misses": 1}
        return {}

    async def aclose(self):
        return None


def test_normalize_query_preserves_numbers_and_entities():
    assert normalize_query("  What   was Revenue in 2023? ") == "what was revenue in 2023?"
    assert normalize_query("What was revenue in 2023?") != normalize_query("What was revenue in 2024?")


def test_cache_key_isolation_across_dimensions():
    k_base = RedisCacheManager.build_answer_key(1, "groq", "gpt-oss-120b", "onnx", ["doc-a", "doc-b"], "Summary?", 0.7)
    # Different order of document_ids must produce identical key
    k_reordered = RedisCacheManager.build_answer_key(1, "groq", "gpt-oss-120b", "onnx", ["doc-b", "doc-a"], "  summary? ", 0.7)
    assert k_base == k_reordered

    # Any dimension change must yield a distinct key
    assert k_base != RedisCacheManager.build_answer_key(2, "groq", "gpt-oss-120b", "onnx", ["doc-a", "doc-b"], "Summary?", 0.7)
    assert k_base != RedisCacheManager.build_answer_key(1, "ollama", "gpt-oss-120b", "onnx", ["doc-a", "doc-b"], "Summary?", 0.7)
    assert k_base != RedisCacheManager.build_answer_key(1, "groq", "llama3.2", "onnx", ["doc-a", "doc-b"], "Summary?", 0.7)
    assert k_base != RedisCacheManager.build_answer_key(1, "groq", "gpt-oss-120b", "baseline", ["doc-a", "doc-b"], "Summary?", 0.7)
    assert k_base != RedisCacheManager.build_answer_key(1, "groq", "gpt-oss-120b", "onnx", ["doc-c"], "Summary?", 0.7)


@pytest.mark.asyncio
async def test_fail_open_circuit_breaker_on_redis_error(monkeypatch):
    mgr = RedisCacheManager()
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")

    broken_client = MagicMock()
    broken_client.get = AsyncMock(side_effect=ConnectionError("Redis refused"))
    mgr._client = broken_client
    mgr._initialized_url = settings.redis_url

    # Must fail-open returning None instead of raising, and trip circuit breaker
    res = await mgr.get_cached_answer("groq", "gpt-oss-120b", "onnx", None, "Hello?")
    assert res is None
    assert mgr._is_circuit_open()

    # Subsequent calls while circuit is open must not even touch the broken client
    broken_client.get.reset_mock()
    res2 = await mgr.get_cached_answer("groq", "gpt-oss-120b", "onnx", None, "Hello?")
    assert res2 is None
    broken_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_embedding_pack_unpack_roundtrip(monkeypatch):
    mgr = RedisCacheManager()
    fake = InMemoryFakeRedis()
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")
    mgr._client = fake
    mgr._initialized_url = settings.redis_url

    vec = [0.125, -0.5, 0.75, 0.001234] * 192  # 768 dims
    await mgr.set_cached_embedding("nomic-embed-text", "What is TARK Studio?", vec)
    loaded = await mgr.get_cached_embedding("nomic-embed-text", "What is TARK Studio?")
    assert loaded is not None
    assert len(loaded) == 768
    assert loaded[0] == pytest.approx(0.125, rel=1e-6)
    assert loaded[3] == pytest.approx(0.001234, rel=1e-6)


@pytest.mark.asyncio
async def test_corpus_version_invalidation(monkeypatch):
    mgr = RedisCacheManager()
    fake = InMemoryFakeRedis()
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")
    mgr._client = fake
    mgr._initialized_url = settings.redis_url

    await mgr.set_cached_answer("groq", "gpt-oss-120b", "onnx", ["doc-1"], "Q1 revenue?", "10M", [{"page": 1}])
    hit1 = await mgr.get_cached_answer("groq", "gpt-oss-120b", "onnx", ["doc-1"], "q1 revenue?")
    assert hit1 is not None and hit1["answer"] == "10M"

    # Simulate document upload/delete -> O(1) version bump invalidates prior cached answer
    new_ver = await mgr.bump_corpus_version()
    assert new_ver == 1
    assert fake.store[CORPUS_VERSION_KEY] == b"1"
    hit2 = await mgr.get_cached_answer("groq", "gpt-oss-120b", "onnx", ["doc-1"], "Q1 revenue?")
    assert hit2 is None


def _parse_sse_events(raw_chunks: list[str]) -> tuple[list[str], str, dict]:
    events = []
    tokens = []
    done_data = {}
    buffer = "".join(raw_chunks)
    while "\n\n" in buffer:
        block, buffer = buffer.split("\n\n", 1)
        ev_type, data_str = "", ""
        for line in block.strip().split("\n"):
            if line.startswith("event: "):
                ev_type = line.split("event: ", 1)[1].strip()
            elif line.startswith("data: "):
                data_str = line.split("data: ", 1)[1].strip()
        if ev_type:
            events.append(ev_type)
        if ev_type == "token" and data_str:
            tokens.append(json.loads(data_str).get("delta", ""))
        elif ev_type == "done" and data_str:
            done_data = json.loads(data_str)
    return events, "".join(tokens), done_data


@pytest.mark.asyncio
async def test_chat_stream_cache_hit_regenerate_and_multiturn_bypass(test_client: AsyncClient, monkeypatch):
    fake = InMemoryFakeRedis()
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")
    redis_cache._client = fake
    redis_cache._initialized_url = settings.redis_url
    redis_cache._cooldown_until = 0.0

    call_count = 0

    async def fake_stream_workflow(**kwargs):
        nonlocal call_count
        call_count += 1
        yield f'event: sources\ndata: {json.dumps({"citations": [{"document_name": "doc.pdf", "viewer_page": 2}], "rag_status": {"retrieval": "Hybrid"}})}\n\n'
        yield f'event: meta\ndata: {json.dumps({"model": "openai/gpt-oss-120b", "provider": "groq", "conversation_id": kwargs.get("conversation_id")})}\n\n'
        yield f'event: token\ndata: {json.dumps({"delta": f"Answer #{call_count}"})}\n\n'
        yield f'event: done\ndata: {json.dumps({"status": "completed", "finish_reason": "stop"})}\n\n'

    doc_uuid = str(uuid.uuid4())

    with patch("app.api.chat.stream_chat_workflow", side_effect=fake_stream_workflow), \
         patch("app.api.chat._persist_user_turn", new=AsyncMock()), \
         patch("app.api.chat.async_session_factory", side_effect=RuntimeError("skip db in unit test")):
        # 1. First request -> Cache MISS, runs workflow, writes behind to Redis
        chunks_1 = []
        async with test_client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={
                "messages": [{"role": "user", "content": "What is Q3 growth?"}],
                "model": "openai/gpt-oss-120b",
                "document_ids": [doc_uuid],
                "temperature": 0.7,
            },
        ) as resp:
            assert resp.status_code == 200
            async for txt in resp.aiter_text():
                chunks_1.append(txt)

        await asyncio.sleep(0.05)  # allow write-behind task to complete
        ev1, ans1, done1 = _parse_sse_events(chunks_1)
        assert call_count == 1
        assert ans1 == "Answer #1"
        assert "sources" in ev1 and "meta" in ev1 and "done" in ev1
        assert done1.get("cached") is not True

        # 2. Repeat normalized query -> Cache HIT, workflow MUST NOT run
        chunks_2 = []
        async with test_client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={
                "messages": [{"role": "user", "content": "  what is q3   growth? "}],
                "model": "openai/gpt-oss-120b",
                "document_ids": [doc_uuid],
                "temperature": 0.7,
            },
        ) as resp:
            assert resp.status_code == 200
            async for txt in resp.aiter_text():
                chunks_2.append(txt)

        ev2, ans2, done2 = _parse_sse_events(chunks_2)
        assert call_count == 1  # Zero LLM/RAG invocation on cache hit!
        assert ans2 == "Answer #1"
        assert "sources" in ev2 and "meta" in ev2 and "token" in ev2 and "done" in ev2
        assert done2.get("cached") is True

        # 3. Retry / Regenerate (save_user_message=False) -> MUST bypass cache & overwrite
        chunks_3 = []
        async with test_client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={
                "messages": [{"role": "user", "content": "What is Q3 growth?"}],
                "model": "openai/gpt-oss-120b",
                "document_ids": [doc_uuid],
                "temperature": 0.7,
                "save_user_message": False,
            },
        ) as resp:
            assert resp.status_code == 200
            async for txt in resp.aiter_text():
                chunks_3.append(txt)

        await asyncio.sleep(0.05)
        _, ans3, _ = _parse_sse_events(chunks_3)
        assert call_count == 2
        assert ans3 == "Answer #2"

        # 4. Subsequent hit must now serve updated Answer #2
        chunks_4 = []
        async with test_client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={
                "messages": [{"role": "user", "content": "What is Q3 growth?"}],
                "model": "openai/gpt-oss-120b",
                "document_ids": [doc_uuid],
                "temperature": 0.7,
            },
        ) as resp:
            async for txt in resp.aiter_text():
                chunks_4.append(txt)
        _, ans4, done4 = _parse_sse_events(chunks_4)
        assert call_count == 2
        assert ans4 == "Answer #2"
        assert done4.get("cached") is True

        # 5. Multi-turn novel follow-up -> MUST bypass cache completely
        async with test_client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={
                "messages": [
                    {"role": "user", "content": "What is Q3 growth?"},
                    {"role": "assistant", "content": "Answer #2"},
                    {"role": "user", "content": "Explain the margin driver in detail"},
                ],
                "model": "openai/gpt-oss-120b",
                "document_ids": [doc_uuid],
                "temperature": 0.7,
            },
        ) as resp:
            async for _ in resp.aiter_text():
                pass
        assert call_count == 3

        # 6. Multi-turn exact repeat of an earlier user query in same chat -> MUST HIT cache
        chunks_6 = []
        async with test_client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={
                "messages": [
                    {"role": "user", "content": "What is Q3 growth?"},
                    {"role": "assistant", "content": "Answer #2"},
                    {"role": "user", "content": "What is Q3 growth?"},
                ],
                "model": "openai/gpt-oss-120b",
                "document_ids": [doc_uuid],
                "temperature": 0.7,
            },
        ) as resp:
            async for txt in resp.aiter_text():
                chunks_6.append(txt)
        _, ans6, done6 = _parse_sse_events(chunks_6)
        assert call_count == 3
        assert ans6 == "Answer #2"
        assert done6.get("cached") is True

    redis_cache._client = None
    redis_cache._initialized_url = None


def test_structural_guard_rules():
    # True paraphrases with matching invariants must pass
    assert passes_structural_guard("what is first task", "tell me the first task") is True
    assert passes_structural_guard("what is first task", "what is the 1st task?") is True
    assert passes_structural_guard("What was Apple revenue in 2023?", "Tell me Apple's 2023 revenue") is True

    # False-positive traps must be deterministically blocked
    assert passes_structural_guard("what is first task", "what is second task") is False
    assert passes_structural_guard("What was Apple revenue in 2023?", "What was Apple revenue in 2024?") is False
    assert passes_structural_guard("is redis enabled in production", "is redis not enabled in production") is False
    assert passes_structural_guard("What is Apple revenue?", "What is Google revenue?") is False
    assert passes_structural_guard('Explain "token budget"', 'Explain "chunk size"') is False


@pytest.mark.asyncio
async def test_semantic_cache_paraphrase_hit_and_guard_block(monkeypatch):
    mgr = RedisCacheManager()
    fake = InMemoryFakeRedis()
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")
    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    monkeypatch.setattr(settings, "semantic_cache_threshold", 0.90)
    mgr._client = fake
    mgr._initialized_url = settings.redis_url

    # Deterministic 4D embeddings: paraphrases share direction, unrelated is orthogonal
    vec_task_1 = [1.0, 0.0, 0.0, 0.0]
    vec_task_1_para = [0.96, 0.28, 0.0, 0.0]  # cosine = 0.96 >= 0.90
    vec_trap_neg = [0.99, 0.14, 0.0, 0.0]     # high cosine 0.99, must STILL be blocked by guard!
    vec_other = [0.0, 0.0, 1.0, 0.0]          # cosine = 0.0

    vec_map = {
        "what is first task": vec_task_1,
        "tell me the first task": vec_task_1_para,
        "what is not first task": vec_trap_neg,
        "what is second task": vec_trap_neg,
        "summarize the whole pdf": vec_other,
    }
    embed_calls = 0

    async def fake_embed(q: str):
        nonlocal embed_calls
        embed_calls += 1
        return vec_map[q]

    # Empty semantic bucket must NOT invoke embedding provider
    miss_empty = await mgr.get_semantic_cached_answer(
        "groq", "gpt-oss-120b", "onnx", ["doc-1"], "tell me the first task", 0.7, fake_embed
    )
    assert miss_empty is None
    assert embed_calls == 0

    # Store original query + vector
    await mgr.set_cached_answer(
        "groq", "gpt-oss-120b", "onnx", ["doc-1"],
        "what is first task", "Task 1 is registration.", [{"page": 1}], 0.7,
        get_embedding_fn=fake_embed,
    )
    assert embed_calls == 1

    # Paraphrase hit!
    hit = await mgr.get_semantic_cached_answer(
        "groq", "gpt-oss-120b", "onnx", ["doc-1"], "tell me the first task", 0.7, fake_embed
    )
    assert hit is not None
    assert hit["answer"] == "Task 1 is registration."
    assert hit["cache_tier"] == "semantic"
    assert hit["similarity"] >= 0.95

    # Trap 1: negation flip blocked despite 0.99 cosine
    assert await mgr.get_semantic_cached_answer(
        "groq", "gpt-oss-120b", "onnx", ["doc-1"], "what is not first task", 0.7, fake_embed
    ) is None

    # Trap 2: ordinal change blocked despite 0.99 cosine
    assert await mgr.get_semantic_cached_answer(
        "groq", "gpt-oss-120b", "onnx", ["doc-1"], "what is second task", 0.7, fake_embed
    ) is None

    # Unrelated query blocked by cosine threshold
    assert await mgr.get_semantic_cached_answer(
        "groq", "gpt-oss-120b", "onnx", ["doc-1"], "summarize the whole pdf", 0.7, fake_embed
    ) is None


@pytest.mark.asyncio
async def test_chat_stream_semantic_paraphrase_hit(test_client: AsyncClient, monkeypatch):
    fake = InMemoryFakeRedis()
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")
    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    monkeypatch.setattr(settings, "semantic_cache_threshold", 0.90)
    redis_cache._client = fake
    redis_cache._initialized_url = settings.redis_url
    redis_cache._cooldown_until = 0.0

    call_count = 0

    async def fake_stream_workflow(**kwargs):
        nonlocal call_count
        call_count += 1
        yield f'event: sources\ndata: {json.dumps({"citations": [{"document_name": "assign.pdf", "viewer_page": 1}], "rag_status": {"retrieval": "Hybrid"}})}\n\n'
        yield f'event: meta\ndata: {json.dumps({"model": "openai/gpt-oss-120b", "provider": "groq", "conversation_id": kwargs.get("conversation_id")})}\n\n'
        yield f'event: token\ndata: {json.dumps({"delta": "Candidate portal task"})}\n\n'
        yield f'event: done\ndata: {json.dumps({"status": "completed", "finish_reason": "stop"})}\n\n'

    async def fake_get_embedding(text: str, is_query: bool = True):
        if "first" in text.lower() or "1st" in text.lower():
            return [0.98, 0.20, 0.0, 0.0] if "tell" in text.lower() else [1.0, 0.0, 0.0, 0.0]
        return [0.99, 0.14, 0.0, 0.0]  # high similarity for "second" too -> guard must block!

    doc_uuid = str(uuid.uuid4())
    with patch("app.api.chat.stream_chat_workflow", side_effect=fake_stream_workflow), \
         patch("app.api.chat._persist_user_turn", new=AsyncMock()), \
         patch("app.api.chat.async_session_factory", side_effect=RuntimeError("skip db")), \
         patch("app.api.chat.embeddings_client.get_embedding", side_effect=fake_get_embedding):

        # 1. Cold query
        async with test_client.stream("POST", "/api/v1/chat/stream", json={
            "messages": [{"role": "user", "content": "what is first task"}],
            "model": "openai/gpt-oss-120b",
            "document_ids": [doc_uuid],
            "temperature": 0.7,
        }) as r1:
            async for _ in r1.aiter_text():
                pass
        await asyncio.sleep(0.05)
        assert call_count == 1

        # 2. Paraphrase in New Chat -> Semantic Cache HIT!
        chunks_2 = []
        async with test_client.stream("POST", "/api/v1/chat/stream", json={
            "messages": [{"role": "user", "content": "tell me the first task"}],
            "model": "openai/gpt-oss-120b",
            "document_ids": [doc_uuid],
            "temperature": 0.7,
        }) as r2:
            async for t in r2.aiter_text():
                chunks_2.append(t)
        _, ans2, done2 = _parse_sse_events(chunks_2)
        assert call_count == 1
        assert ans2 == "Candidate portal task"
        assert done2.get("cached") is True
        assert done2.get("cache_tier") == "semantic"

        # 3. Ordinal swap ("second task") -> MUST miss and call workflow!
        async with test_client.stream("POST", "/api/v1/chat/stream", json={
            "messages": [{"role": "user", "content": "what is second task"}],
            "model": "openai/gpt-oss-120b",
            "document_ids": [doc_uuid],
            "temperature": 0.7,
        }) as r3:
            async for _ in r3.aiter_text():
                pass
        assert call_count == 2

    redis_cache._client = None
    redis_cache._initialized_url = None


@pytest.mark.asyncio
async def test_health_exposes_safe_redis_stats(test_client: AsyncClient, monkeypatch):
    fake = InMemoryFakeRedis()
    fake.store["tark:ans:v0:k1"] = b'{"answer": "secret"}'
    fake.store["tark:emb:m1:k2"] = b"1234"
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")
    redis_cache._client = fake
    redis_cache._initialized_url = settings.redis_url
    redis_cache._cooldown_until = 0.0

    with patch("app.main.check_db_connectivity", new=AsyncMock(return_value={"connected": True})):
        resp = await test_client.get("/api/v1/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["redis"]["connected"] is True
    assert body["redis"]["answer_keys"] == 1
    assert body["redis"]["embedding_keys"] == 1
    assert body["redis"]["used_memory_human"] == "1.00M"
    assert body["redis"]["hits"] == 3
    assert body["redis"]["misses"] == 1
    assert "secret" not in resp.text

    redis_cache._client = None
    redis_cache._initialized_url = None
