"""Rate-limit aware request tracking and per-provider queue for TARK Studio."""

import asyncio
import logging
import re
import time
from typing import Callable, Optional, Dict, Any, List

logger = logging.getLogger(__name__)


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
    """Rate-limit aware request tracker with proactive token-bucket tracking and controlled queue."""

    def __init__(self, provider: str = "groq", default_limit_tokens: int = 8000):
        self.provider = provider
        self.limit_tokens = default_limit_tokens
        self.remaining_tokens = default_limit_tokens
        self.reset_tokens_sec = 0.0
        self.last_updated = time.perf_counter()
        self.lock = asyncio.Lock()
        self.last_queue_wait_ms = 0.0
        self.last_headers: Dict[str, Any] = {}

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
