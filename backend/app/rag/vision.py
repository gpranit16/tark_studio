"""Cloudflare Workers AI Vision Client for Scanned Document OCR.

Provides asynchronous, rate-limit resilient transcription for scanned document pages
using Cloudflare Workers AI Llama-3.2 Vision.
"""

import asyncio
import logging
import random
import time
from typing import Optional
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

CLOUDFLARE_DEFAULT_VISION_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"


class CloudflareVisionClient:
    """Non-blocking, resilient client for Cloudflare Workers AI Vision inference.
    
    Includes live HTTP 429 / Retry-After inspection and decorrelated jitter backoff.
    """

    def __init__(self):
        self.account_id = settings.cloudflare_account_id
        self.api_token = settings.cloudflare_api_token
        self.model_name = getattr(settings, "cloudflare_vision_model", CLOUDFLARE_DEFAULT_VISION_MODEL)
        self.timeout_sec = 25.0
        self.max_retries = 2

    @property
    def is_configured(self) -> bool:
        """Check whether Cloudflare Workers AI credentials are validly provisioned."""
        return bool(self.account_id and self.api_token and getattr(settings, "enable_vision_ocr", True))

    async def transcribe_image_bytes(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Transcribe text, tables, numbers, and signatures from image bytes verbatim."""
        if not self.is_configured:
            logger.debug("Cloudflare Vision OCR skipped: credentials not configured in environment.")
            return None

        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        default_prompt = (
            "Transcribe all printed and handwritten text, names, dates, amounts, "
            "and table cells from this scanned document page verbatim. "
            "Preserve layout order. Do not summarize, explain, or hallucinate."
        )

        payload = {
            "prompt": prompt or default_prompt,
            "image": list(image_bytes),
            "max_tokens": 1024,
        }

        backoff = 1.0
        for attempt in range(1, self.max_retries + 2):
            try:
                t0 = time.perf_counter()
                async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                
                dur = round(time.perf_counter() - t0, 2)

                if resp.status_code == 200:
                    data = resp.json()
                    result = data.get("result", {})
                    response_text = result.get("response", "") if isinstance(result, dict) else str(result)
                    logger.info(f"Cloudflare Vision OCR completed in {dur}s ({len(response_text)} chars).")
                    return response_text.strip() if response_text else None

                # Handle HTTP 429 Too Many Requests
                if resp.status_code == 429:
                    retry_after_header = resp.headers.get("retry-after")
                    retry_sec = float(retry_after_header) if retry_after_header and retry_after_header.isdigit() else backoff
                    sleep_time = min(6.0, retry_sec + random.uniform(0.1, 0.5))
                    logger.warning(
                        f"Cloudflare Vision 429 Rate Limit (attempt {attempt}/{self.max_retries + 1}). "
                        f"Backing off for {round(sleep_time, 2)}s..."
                    )
                    await asyncio.sleep(sleep_time)
                    backoff = min(6.0, backoff * 2.0)
                    continue

                logger.warning(f"Cloudflare Vision returned HTTP {resp.status_code}: {resp.text[:200]}")
                return None

            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException:
                logger.warning(f"Cloudflare Vision timed out after {self.timeout_sec}s (attempt {attempt}).")
                if attempt <= self.max_retries:
                    await asyncio.sleep(1.0)
                    continue
                return None
            except Exception as exc:
                logger.warning(f"Cloudflare Vision inference exception: {exc}")
                return None

        return None


vision_client = CloudflareVisionClient()
