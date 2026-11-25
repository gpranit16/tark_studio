"""Unit tests for ProviderRateLimiter and rate limit tracking."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.ai.rate_limiter import ProviderRateLimiter, parse_reset_time_seconds
from app.config import settings
from app.rag.cache import redis_cache


def test_parse_reset_time_seconds():
    assert parse_reset_time_seconds("500ms") == pytest.approx(0.5)
    assert parse_reset_time_seconds("18.2s") == pytest.approx(18.2)
    assert parse_reset_time_seconds("1m30s") == pytest.approx(90.0)
    assert parse_reset_time_seconds("2m10.5s") == pytest.approx(130.5)
    assert parse_reset_time_seconds(None) == 0.0
    assert parse_reset_time_seconds("invalid") == 1.0


def test_rate_limiter_update_from_headers():
    limiter = ProviderRateLimiter(provider="test", default_limit_tokens=8000)
    headers = {
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "4500",
        "x-ratelimit-reset-tokens": "12.5s",
    }
    limiter.update_from_headers(headers)
    assert limiter.limit_tokens == 8000
    assert limiter.remaining_tokens == 4500
    assert limiter.reset_tokens_sec == pytest.approx(12.5)


def test_rate_limiter_refill_estimation():
    limiter = ProviderRateLimiter(provider="test", default_limit_tokens=8000)
    limiter.remaining_tokens = 2000
    limiter.last_updated = time.perf_counter() - 3.0  # 3 seconds ago
    # Refill rate = 8000 / 60 = 133.33 tok/s -> in 3s should refill ~400 tokens
    estimated = limiter.get_estimated_remaining()
    assert estimated >= 2390
    assert estimated <= 2450


def test_rate_limiter_calculate_wait_seconds():
    limiter = ProviderRateLimiter(provider="test", default_limit_tokens=8000)
    limiter.remaining_tokens = 5000
    limiter.last_updated = time.perf_counter()
    # Prompt of 3000 tokens needs 0 wait
    assert limiter.calculate_wait_seconds(3000) == 0.0
    # Prompt of 6000 tokens has 1000 deficit -> 1000 / 133.33 = ~7.5s wait
    wait = limiter.calculate_wait_seconds(6000)
    assert wait == pytest.approx(7.5, abs=0.5)


@pytest.mark.asyncio
async def test_rate_limiter_acquire_slot_immediate():
    limiter = ProviderRateLimiter(provider="test", default_limit_tokens=8000)
    limiter.remaining_tokens = 7000
    limiter.last_updated = time.perf_counter()

    status_events = []
    def on_status(ev):
        status_events.append(ev)

    wait_ms = await limiter.acquire_slot(prompt_tokens=2000, on_status=on_status)
    assert wait_ms >= 0.0
    assert len(status_events) == 0
    assert limiter.remaining_tokens <= 5000


class _FakeRedisRL:
    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}
        self.hashes: dict[str, dict] = {}

    async def get(self, key: str):
        return self.kv.get(key)

    async def setex(self, key: str, ttl: int, val: str):
        self.kv[key] = val.encode("utf-8")
        return True

    async def hset(self, key: str, mapping: dict):
        self.hashes[key] = mapping
        return True

    async def expire(self, key: str, ttl: int):
        return True


@pytest.mark.asyncio
async def test_shared_cooldown_propagation_across_workers(monkeypatch):
    fake_redis = _FakeRedisRL()
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")
    redis_cache._client = fake_redis
    redis_cache._initialized_url = settings.redis_url
    redis_cache._cooldown_until = 0.0

    worker1 = ProviderRateLimiter(provider="groq_test", default_limit_tokens=8000)
    worker2 = ProviderRateLimiter(provider="groq_test", default_limit_tokens=8000)

    # Worker 1 hits 429 and broadcasts cooldown -> Worker 2 must see it via Redis
    worker1.record_cooldown(2.0)
    await asyncio.sleep(0.02)

    rem = await worker2.get_shared_cooldown_remaining()
    assert 1.0 < rem <= 2.1

    redis_cache._client = None
    redis_cache._initialized_url = None


@pytest.mark.asyncio
async def test_rate_limiter_fail_open_on_redis_error(monkeypatch):
    monkeypatch.setattr(settings, "redis_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")
    broken = MagicMock()
    broken.get = AsyncMock(side_effect=ConnectionError("Redis offline"))
    broken.hset = AsyncMock(side_effect=ConnectionError("Redis offline"))
    redis_cache._client = broken
    redis_cache._initialized_url = settings.redis_url
    redis_cache._cooldown_until = 0.0

    limiter = ProviderRateLimiter(provider="groq_failopen", default_limit_tokens=8000)
    # Must never raise on Redis failure
    limiter.update_from_headers({"x-ratelimit-remaining-tokens": "3000"})
    await asyncio.sleep(0.02)
    wait_ms = await limiter.wait_for_shared_cooldown()
    assert wait_ms == 0.0
    assert limiter.remaining_tokens == 3000

    redis_cache._client = None
    redis_cache._initialized_url = None
