"""Rate-limit aware request tracking and fail-open distributed Redis coordination for TARK Studio."""

import asyncio
import logging
import math
import re
import time
from typing import Callable, Optional, Dict, Any, List

from app.rag.cache import redis_cache

logger = logging.getLogger(__name__)
_rl_sync_tasks: set[asyncio.Task] = set()
_MAX_SHARED_COOLDOWN_SEC = 10.0


def _schedule_bg(coro) -> None:
    """Schedule async Redis sync if an event loop is active; retain strong reference."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        coro.close()
        return
    task = asyncio.create_task(coro)
    _rl_sync_tasks.add(task)
    task.add_done_callback(_rl_sync_tasks.discard)


def parse_reset_time_seconds(reset_str: Optional[str]) -> float:
    """Parse Groq reset time format like '547ms', '18.2s', '1m26.4s' into seconds."""
    if not reset_str:
        return 0.0
    s = reset_str.strip().lower()
    try:
        if s.endswith("ms"):
            return float(s[:-2]) / 1000.0
        if "m" in s and s.endswith("s"):
            parts = s.split("m")
            mins = float(parts[0])
            secs = float(parts[1][:-1]) if parts[1].endswith("s") else float(parts[1])
            return mins * 60.0 + secs
        if s.endswith("s"):
            return float(s[:-1])
        return float(s)
    except Exception:
        return 1.0


class ProviderRateLimiter:
    """Rate-limit aware request tracker with fail-open Redis multi-worker coordination."""

    def __init__(self, provider: str = "groq", default_limit_tokens: int = 8000):
        self.provider = provider
        self.limit_tokens = default_limit_tokens
        self.remaining_tokens = default_limit_tokens
        self.reset_tokens_sec = 0.0
        self.last_updated = time.perf_counter()
        self.lock = asyncio.Lock()
        self.last_queue_wait_ms = 0.0
        self.last_headers: Dict[str, Any] = {}
        self._cooldown_until_wall: float = 0.0

    @property
    def _state_key(self) -> str:
        return f"tark:rl:{self.provider}:state"

    @property
    def _cooldown_key(self) -> str:
        return f"tark:rl:{self.provider}:cooldown"

    async def _sync_state_to_redis(self) -> None:
        client = await redis_cache.get_client()
        if client is None:
            return
        try:
            await client.hset(
                self._state_key,
                mapping={
                    "limit_tokens": str(self.limit_tokens),
                    "remaining_tokens": str(self.remaining_tokens),
                    "updated_wall": str(time.time()),
                },
            )
            await client.expire(self._state_key, 120)
        except Exception as exc:
            logger.debug(f"[{self.provider}] Redis rate-limit state sync skipped (fail-open): {exc}")

    async def _sync_cooldown_to_redis(self, cooldown_until: float, ttl_sec: int) -> None:
        client = await redis_cache.get_client()
        if client is None:
            return
        try:
            await client.setex(self._cooldown_key, max(1, ttl_sec), str(cooldown_until))
        except Exception as exc:
            logger.debug(f"[{self.provider}] Redis cooldown sync skipped (fail-open): {exc}")

    def record_cooldown(self, retry_after_sec: float) -> None:
        """Broadcast a provider 429 cooldown across workers without blocking callers."""
        bounded = min(_MAX_SHARED_COOLDOWN_SEC, max(0.5, float(retry_after_sec)))
        cooldown_until = time.time() + bounded
        self._cooldown_until_wall = max(self._cooldown_until_wall, cooldown_until)
        if redis_cache.is_enabled:
            _schedule_bg(self._sync_cooldown_to_redis(cooldown_until, int(math.ceil(bounded))))

    async def get_shared_cooldown_remaining(self) -> float:
        """Return remaining seconds on shared cooldown (0.0 if clear or Redis unavailable)."""
        now = time.time()
        local_rem = max(0.0, self._cooldown_until_wall - now)

        if not redis_cache.is_enabled:
            return min(_MAX_SHARED_COOLDOWN_SEC, local_rem)

        client = await redis_cache.get_client()
        if client is None:
            return min(_MAX_SHARED_COOLDOWN_SEC, local_rem)

        try:
            raw = await client.get(self._cooldown_key)
            if raw is not None:
                remote_until = float(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                self._cooldown_until_wall = max(self._cooldown_until_wall, remote_until)
                local_rem = max(local_rem, remote_until - now)
        except Exception as exc:
            logger.debug(f"[{self.provider}] Redis cooldown read skipped (fail-open): {exc}")

        return min(_MAX_SHARED_COOLDOWN_SEC, max(0.0, local_rem))

    async def wait_for_shared_cooldown(
        self,
        on_status: Optional[Callable[[dict], None]] = None,
        request=None,
    ) -> float:
        """Wait briefly only if another worker recently triggered a 429 cooldown; 0ms on healthy path."""
        wait_sec = await self.get_shared_cooldown_remaining()
        if wait_sec <= 0.1:
            return 0.0

        t_start = time.perf_counter()
        remaining = wait_sec
        step = 0.5
        while remaining > 0:
            if request and await request.is_disconnected():
                raise asyncio.CancelledError()
            if on_status:
                on_status({
                    "status": "waiting",
                    "reason": "Shared provider cooldown active",
                    "wait_sec": round(remaining, 1),
                    "remaining_tokens": 0,
                    "limit_tokens": self.limit_tokens,
                })
            chunk = min(step, remaining)
            await asyncio.sleep(chunk)
            remaining -= chunk

        if on_status:
            on_status({
                "status": "ready",
                "reason": "Shared cooldown cleared",
                "wait_sec": 0.0,
                "remaining_tokens": self.get_estimated_remaining(),
                "limit_tokens": self.limit_tokens,
            })
        return round((time.perf_counter() - t_start) * 1000, 1)

    def update_from_headers(self, headers: Any):
        """Update live quota state from HTTP response headers."""
        if not headers:
            return
        try:
            h_get = getattr(headers, "get", None)
            if not callable(h_get):
                return

            limit_t = h_get("x-ratelimit-limit-tokens")
            rem_t = h_get("x-ratelimit-remaining-tokens")
            reset_t = h_get("x-ratelimit-reset-tokens")

            if limit_t is not None:
                try:
                    self.limit_tokens = int(limit_t)
                except ValueError:
                    pass

            if rem_t is not None:
                try:
                    self.remaining_tokens = int(rem_t)
                except ValueError:
                    pass

            if reset_t is not None:
                self.reset_tokens_sec = parse_reset_time_seconds(str(reset_t))

            self.last_updated = time.perf_counter()
            self.last_headers = {
                "limit_tokens": self.limit_tokens,
                "remaining_tokens": self.remaining_tokens,
                "reset_tokens_sec": self.reset_tokens_sec,
            }
            logger.info(
                f"[{self.provider}] Rate limiter updated: remaining={self.remaining_tokens}/{self.limit_tokens}, "
                f"reset={round(self.reset_tokens_sec, 2)}s"
            )
            if redis_cache.is_enabled:
                _schedule_bg(self._sync_state_to_redis())
        except Exception as exc:
            logger.warning(f"Error parsing rate limit headers: {exc}")

    def get_estimated_remaining(self) -> int:
        """Estimate tokens currently available after continuous refill."""
        elapsed = time.perf_counter() - self.last_updated
        refill_rate = self.limit_tokens / 60.0
        refilled = int(elapsed * refill_rate)
        return min(self.limit_tokens, self.remaining_tokens + refilled)

    def calculate_wait_seconds(self, prompt_tokens: int) -> float:
        """Calculate seconds required before prompt fits into available token budget."""
        available = self.get_estimated_remaining()
        if available >= prompt_tokens:
            return 0.0
        deficit = prompt_tokens - available
        refill_rate = self.limit_tokens / 60.0
        # Cap wait at 30s to prevent unbounded stalls
        return min(30.0, max(0.5, deficit / refill_rate))

    async def acquire_slot(
        self,
        prompt_tokens: int,
        on_status: Optional[Callable[[dict], None]] = None,
        request=None,
    ) -> float:
        """Controlled queue: waits until prompt fits token bucket, emitting progress updates."""
        t_start = time.perf_counter()

        async with self.lock:
            wait_sec = self.calculate_wait_seconds(prompt_tokens)
            if wait_sec > 0.5:
                logger.info(
                    f"[{self.provider}] Proactive TPM rate pacing: prompt needs {prompt_tokens}t, "
                    f"available {self.get_estimated_remaining()}t. Waiting {round(wait_sec, 1)}s..."
                )
                remaining_wait = wait_sec
                step = 0.5
                while remaining_wait > 0:
                    if request and await request.is_disconnected():
                        logger.info(f"[{self.provider}] Client disconnected during rate-limit queue wait.")
                        raise asyncio.CancelledError()

                    if on_status:
                        on_status({
                            "status": "waiting",
                            "reason": "TPM token budget refill",
                            "wait_sec": round(remaining_wait, 1),
                            "remaining_tokens": self.get_estimated_remaining(),
                            "limit_tokens": self.limit_tokens,
                        })

                    sleep_chunk = min(step, remaining_wait)
                    await asyncio.sleep(sleep_chunk)
                    remaining_wait -= sleep_chunk

                if on_status:
                    on_status({
                        "status": "ready",
                        "reason": "Token budget available",
                        "wait_sec": 0.0,
                        "remaining_tokens": self.get_estimated_remaining(),
                        "limit_tokens": self.limit_tokens,
                    })

            # Deduct tokens from estimated balance
            current = self.get_estimated_remaining()
            self.remaining_tokens = max(0, current - prompt_tokens)
            self.last_updated = time.perf_counter()

            total_wait_ms = round((time.perf_counter() - t_start) * 1000, 1)
            self.last_queue_wait_ms = total_wait_ms
            return total_wait_ms


# Global rate limiter singleton for Groq
groq_rate_limiter = ProviderRateLimiter(provider="groq", default_limit_tokens=8000)
