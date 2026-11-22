"""Unit tests for ProviderRateLimiter and rate limit tracking."""

import time
import pytest
from app.ai.rate_limiter import ProviderRateLimiter, parse_reset_time_seconds


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
