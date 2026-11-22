"""Cloudflare Workers AI Vision & Local OCR Client for Scanned Document Ingestion.

Provides asynchronous, resilient transcription for scanned document pages using:
1. Cloudflare Workers AI Llama-3.2 Vision (@cf/meta/llama-3.2-11b-vision-instruct)
2. Local Tesseract OCR (if installed in OS/Docker container)

Includes automatic Meta license agreement negotiation, 429 Retry-After handling,
and graceful fallback without blocking digital document ingestion.
"""

import asyncio
import logging
import random
import shutil
import time
from typing import Optional
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

CLOUDFLARE_DEFAULT_VISION_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"


class CloudflareVisionClient:
    """Non-blocking, resilient client for Cloudflare Workers AI Vision inference."""

    def __init__(self):
        self.account_id = settings.cloudflare_account_id
        self.api_token = settings.cloudflare_api_token
        self.model_name = getattr(
            settings, "cloudflare_vision_model", CLOUDFLARE_DEFAULT_VISION_MODEL
        )
        self.timeout_sec = 18.0
        self.max_retries = 2

    @property
    def is_configured(self) -> bool:
        """Check whether Cloudflare Workers AI or local OCR is available."""
        if not getattr(settings, "enable_vision_ocr", True):
            return False
        has_cf = bool(self.account_id and self.api_token)
        has_tesseract = shutil.which("tesseract") is not None
        return has_cf or has_tesseract

    async def transcribe_image_bytes(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Transcribe text, tables, numbers, and signatures from image bytes verbatim."""
        if not self.is_configured:
            logger.debug("Vision OCR skipped: credentials not configured in environment.")
            return None

        default_prompt = (
            "Transcribe all printed and handwritten text, names, dates, amounts, "
            "clauses, and table cells from this scanned document page verbatim in layout order. "
            "Output ONLY the raw text. Do not summarize, explain, or add conversational preamble."
        )
        active_prompt = prompt or default_prompt

        # 1. Primary: Cloudflare Workers AI Llama-3.2 Vision
        if self.account_id and self.api_token:
            res = await self._transcribe_cloudflare(image_bytes, active_prompt)
            if res:
                return res

        # 2. Secondary: Local Tesseract OCR (if available in Linux Docker)
        if shutil.which("tesseract"):
            res = await self._transcribe_tesseract(image_bytes)
            if res:
                return res

        return None

    async def _transcribe_cloudflare(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """Transcribe via Cloudflare Workers AI with license handling and retry."""
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"
            f"/ai/run/{self.model_name}"
        )
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "image": list(image_bytes),
            "max_tokens": 1024,
        }

        backoff = 0.8
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

                # First-time Meta license agreement check
                resp_txt = resp.text.lower()
                if resp.status_code in (400, 403) and any(w in resp_txt for w in ("agree", "terms", "license")):
                    logger.info("Cloudflare Llama-3.2 Vision license terms required. Sending agreement payload...")
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.post(url, json={"prompt": "agree"}, headers=headers)
                    # Retry with payload immediately
                    async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                        retry_resp = await client.post(url, json=payload, headers=headers)
                    if retry_resp.status_code == 200:
                        data = retry_resp.json()
                        result = data.get("result", {})
                        response_text = result.get("response", "") if isinstance(result, dict) else str(result)
                        return response_text.strip() if response_text else None

                # Handle HTTP 429 Too Many Requests with tight backoff
                if resp.status_code == 429:
                    retry_after_header = resp.headers.get("retry-after")
                    retry_sec = float(retry_after_header) if retry_after_header and retry_after_header.isdigit() else backoff
                    sleep_time = min(2.5, retry_sec + random.uniform(0.1, 0.3))
                    logger.warning(
                        f"Cloudflare Vision 429 Rate Limit (attempt {attempt}/{self.max_retries + 1}). "
                        f"Backing off for {round(sleep_time, 2)}s..."
                    )
                    await asyncio.sleep(sleep_time)
                    backoff = min(3.0, backoff * 1.5)
                    continue

                logger.warning(f"Cloudflare Vision returned HTTP {resp.status_code}: {resp.text[:200]}")
                return None

            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException:
                logger.warning(f"Cloudflare Vision timed out after {self.timeout_sec}s (attempt {attempt}).")
                if attempt <= self.max_retries:
                    await asyncio.sleep(0.5)
                    continue
                return None
            except Exception as exc:
                logger.warning(f"Cloudflare Vision inference exception: {exc}")
                if attempt <= self.max_retries:
                    await asyncio.sleep(0.5)
                    continue
                return None

        return None

    async def _transcribe_tesseract(self, image_bytes: bytes) -> Optional[str]:
        """Transcribe via local Tesseract OCR if installed in OS/container."""
        tess_path = shutil.which("tesseract")
        if not tess_path:
            return None
        import subprocess

        try:
            t0 = time.perf_counter()
            proc = await asyncio.to_thread(
                subprocess.run,
                [tess_path, "stdin", "stdout", "--oem", "1", "-l", "eng"],
                input=image_bytes,
                capture_output=True,
                timeout=15,
            )
            dur = round(time.perf_counter() - t0, 2)
            if proc.returncode == 0:
                text = proc.stdout.decode("utf-8", errors="replace").strip()
                if text:
                    logger.info(f"Local Tesseract OCR completed in {dur}s ({len(text)} chars).")
                    return text
        except Exception as exc:
            logger.warning(f"Local Tesseract OCR failed: {exc}")
        return None


# Backward-compatible alias
UnifiedVisionOCRClient = CloudflareVisionClient

vision_client = CloudflareVisionClient()
