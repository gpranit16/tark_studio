"""Unified Vision & OCR Client for Scanned Document Ingestion.

Provides asynchronous, resilient, multi-provider transcription for scanned document pages
supporting:
1. Google Gemini Flash Vision (if GEMINI_API_KEY is configured in environment)
2. Cloudflare Workers AI Llama-3.2 Vision (if CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN configured)
3. Local Tesseract OCR (if installed in OS/Docker container)

Includes automatic license agreement negotiation, 429 Retry-After handling,
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


class UnifiedVisionOCRClient:
    """Non-blocking, resilient client for document Vision OCR inference."""

    def __init__(self):
        self.gemini_api_key = settings.gemini_api_key
        self.cloudflare_account_id = settings.cloudflare_account_id
        self.cloudflare_api_token = settings.cloudflare_api_token
        self.cloudflare_vision_model = getattr(
            settings, "cloudflare_vision_model", CLOUDFLARE_DEFAULT_VISION_MODEL
        )
        self.timeout_sec = 25.0
        self.max_retries = 2

    @property
    def is_configured(self) -> bool:
        """Check whether at least one Vision/OCR engine is configured and enabled."""
        if not getattr(settings, "enable_vision_ocr", True):
            return False
        has_gemini = bool(self.gemini_api_key)
        has_cf = bool(self.cloudflare_account_id and self.cloudflare_api_token)
        has_tesseract = shutil.which("tesseract") is not None
        return has_gemini or has_cf or has_tesseract

    async def transcribe_image_bytes(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Transcribe text, tables, numbers, and signatures from image bytes verbatim."""
        if not self.is_configured:
            logger.debug("Vision OCR skipped: no provider credentials configured in environment.")
            return None

        default_prompt = (
            "Transcribe all printed and handwritten text, names, dates, amounts, "
            "and table cells from this scanned document page verbatim. "
            "Preserve layout order. Do not summarize, explain, or hallucinate."
        )
        active_prompt = prompt or default_prompt

        # 1. Primary: Google Gemini Flash Vision (ultra-low latency, highly accurate)
        if self.gemini_api_key:
            res = await self._transcribe_gemini(image_bytes, active_prompt)
            if res:
                return res

        # 2. Secondary: Cloudflare Workers AI Llama-3.2 Vision
        if self.cloudflare_account_id and self.cloudflare_api_token:
            res = await self._transcribe_cloudflare(image_bytes, active_prompt)
            if res:
                return res

        # 3. Tertiary: Local Tesseract OCR (if available in Linux Docker)
        if shutil.which("tesseract"):
            res = await self._transcribe_tesseract(image_bytes)
            if res:
                return res

        return None

    async def _transcribe_gemini(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """Transcribe via Google Gemini Flash."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_api_key)
            part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

            t0 = time.perf_counter()
            resp = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[part, prompt],
            )
            dur = round(time.perf_counter() - t0, 2)
            if resp and resp.text:
                text = resp.text.strip()
                logger.info(f"Gemini Vision OCR completed in {dur}s ({len(text)} chars).")
                return text
        except Exception as exc:
            logger.warning(f"Gemini Vision OCR failed: {exc}")
        return None

    async def _transcribe_cloudflare(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """Transcribe via Cloudflare Workers AI with license handling and retry."""
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}"
            f"/ai/run/{self.cloudflare_vision_model}"
        )
        headers = {
            "Authorization": f"Bearer {self.cloudflare_api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
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
                if attempt <= self.max_retries:
                    await asyncio.sleep(1.0)
                    continue
                return None

        return None

    async def _transcribe_tesseract(self, image_bytes: bytes) -> Optional[str]:
        """Transcribe via local Tesseract OCR if installed."""
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


# Backward-compatible alias for existing imports and tests
CloudflareVisionClient = UnifiedVisionOCRClient

vision_client = UnifiedVisionOCRClient()
