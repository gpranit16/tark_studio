"""Model provider abstraction for TARK Studio supporting Groq, Gemini, and Ollama."""

import logging
import re
import asyncio
from typing import AsyncGenerator, List, Optional, Callable
from groq import AsyncGroq, RateLimitError
import ollama

from app.config import settings
from app.ai.rate_limiter import groq_rate_limiter, parse_reset_time_seconds

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = [
    # Groq (Cloud Fast Inference)
    {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B (Groq)", "provider": "groq", "description": "High capability open-weight reasoning model hosted on Groq LPU."},
    {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B (Groq Fast)", "provider": "groq", "description": "Fast low-latency completion model on Groq."},
    {"id": "qwen/qwen3.8-27b", "name": "Qwen 3.8 27B (Groq)", "provider": "groq", "description": "Multilingual reasoning model on Groq."},
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B (Groq)", "provider": "groq", "description": "Meta Llama 3.3 versatile 70B on Groq."},

    # Gemini (Google Cloud)
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "gemini", "description": "Google lightweight multimodal model with low latency."},
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "gemini", "description": "Google flagship multimodal reasoning model."},

    # Ollama (Local First)
    {"id": "llama3.2", "name": "Llama 3.2 (Local Ollama)", "provider": "ollama", "description": "Local offline inference via Ollama runtime."},
    {"id": "qwen2.5", "name": "Qwen 2.5 (Local Ollama)", "provider": "ollama", "description": "Local multilingual model via Ollama runtime."},
]


def resolve_provider(model: str) -> str:
    """Resolve provider ('groq' | 'gemini' | 'ollama') from model ID."""
    for item in SUPPORTED_MODELS:
        if item["id"] == model:
            return item["provider"]
    if "gemini" in model.lower():
        return "gemini"
    if "ollama" in model.lower() or model in ["llama3.2", "qwen2.5", "mistral"]:
        return "ollama"
    return "groq"


def get_groq_client() -> AsyncGroq:
    """Instantiate AsyncGroq client using environment-configured key with max_retries=0.
    
    Disables hidden internal SDK retry sleeps so rate-limit pauses are explicitly managed.
    """
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not configured in environment.")
    return AsyncGroq(api_key=settings.groq_api_key, max_retries=0)


def get_ollama_client() -> ollama.AsyncClient:
    """Instantiate Ollama AsyncClient using configured base URL."""
    return ollama.AsyncClient(host=settings.ollama_base_url)


async def stream_groq(
    model: str,
    messages: List[dict],
    temperature: float,
    writer: Optional[Callable] = None,
    estimated_prompt_tokens: int = 1500,
    request=None,
) -> AsyncGenerator[str, None]:
    """Stream token chunks from Groq with explicit TPM budget tracking and 429 retry handling."""
    client = get_groq_client()
    formatted = [{"role": m["role"], "content": m["content"]} for m in messages]

    def notify_rate_limit(payload: dict):
        if writer:
            writer({"type": "rate_limit", "data": payload})

    max_attempts = 3
    response_stream = None

    for attempt in range(1, max_attempts + 1):
        if request and await request.is_disconnected():
            logger.info("Client disconnected before sending request to Groq.")
            raise asyncio.CancelledError()

        try:
            response_stream = await client.chat.completions.create(
                model=model,
                messages=formatted,
                temperature=temperature,
                stream=True,
            )
            # Live header extraction from httpx.Response attached to AsyncStream
            if hasattr(response_stream, "response") and hasattr(response_stream.response, "headers"):
                groq_rate_limiter.update_from_headers(response_stream.response.headers)
            break

        except RateLimitError as rle:
            logger.warning(f"Groq RateLimitError on attempt {attempt}: {rle}")
            err_str = str(rle)

            # Check if this is a daily token limit (TPD) rather than per-minute limit (TPM)
            if "tokens per day" in err_str.lower() or "(tpd)" in err_str.lower():
                m_time = re.search(r"try again in ([^\.]+)", err_str)
                reset_hint = f" (resets in {m_time.group(1)})" if m_time else ""
                err_msg = f"Groq free daily token allowance (200,000 TPD) reached{reset_hint}. Please try again later or switch models."
                if writer:
                    writer({"type": "error", "data": {"error": err_msg}})
                raise RuntimeError(err_msg)

            # Normal TPM rate limit: extract real retry time or default to 1.5s
            retry_after = 1.5
            if hasattr(rle, "response") and rle.response and hasattr(rle.response, "headers"):
                raw_ra = rle.response.headers.get("retry-after")
                if raw_ra:
                    try:
                        retry_after = float(raw_ra)
                    except ValueError:
                        retry_after = parse_reset_time_seconds(raw_ra)
                groq_rate_limiter.update_from_headers(rle.response.headers)
            else:
                m = re.search(r"try again in ([\d\.]+)s", err_str)
                if m:
                    retry_after = float(m.group(1))

            # Cap TPM retry wait to at most 10 seconds to avoid long hangs
            retry_after = min(10.0, max(0.5, retry_after))

            if attempt == max_attempts:
                if writer:
                    writer({"type": "error", "data": {"error": f"Groq rate limit exceeded after {max_attempts} attempts: {rle}"}})
                raise

            logger.info(f"Explicitly handling Groq TPM 429: waiting {round(retry_after, 1)}s before attempt {attempt + 1}...")
            remaining_ra = retry_after
            while remaining_ra > 0:
                if request and await request.is_disconnected():
                    raise asyncio.CancelledError()
                notify_rate_limit({
                    "status": "rate_limited",
                    "reason": f"Groq rate limit reached. Retrying in {round(remaining_ra, 1)}s...",
                    "wait_sec": round(remaining_ra, 1),
                    "retry_after_sec": round(retry_after, 1),
                    "remaining_tokens": 0,
                    "limit_tokens": groq_rate_limiter.limit_tokens,
                })
                sleep_chunk = min(0.5, remaining_ra)
                await asyncio.sleep(sleep_chunk)
                remaining_ra -= sleep_chunk

            notify_rate_limit({"status": "ready", "reason": "Retrying request now", "wait_sec": 0.0})

    if response_stream is None:
        raise RuntimeError("Failed to establish streaming response with Groq.")

    try:
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    finally:
        try:
            await response_stream.close()
        except Exception:
            pass


async def stream_gemini(
    model: str,
    messages: List[dict],
    temperature: float,
) -> AsyncGenerator[str, None]:
    """Stream token chunks from Google Gemini."""
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured in environment.")

    from google import genai
    client = genai.Client(api_key=settings.gemini_api_key)

    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    response_stream = await client.aio.models.generate_content_stream(
        model=model,
        contents=contents,
        config={"temperature": temperature},
    )

    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text


async def stream_ollama(
    model: str,
    messages: List[dict],
    temperature: float,
) -> AsyncGenerator[str, None]:
    """Stream token chunks from local Ollama instance."""
    client = get_ollama_client()
    formatted = [{"role": m["role"], "content": m["content"]} for m in messages]

    response_stream = await client.chat(
        model=model,
        messages=formatted,
        options={"temperature": temperature},
        stream=True,
    )

    async for chunk in response_stream:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]


async def get_model_stream(
    model: str,
    messages: List[dict],
    temperature: float = 0.7,
    provider: Optional[str] = None,
    writer: Optional[Callable] = None,
    estimated_prompt_tokens: int = 1500,
    request=None,
) -> AsyncGenerator[str, None]:
    """Unified dispatcher streaming token chunks from the selected provider."""
    resolved = provider or resolve_provider(model)

    if resolved == "groq":
        async for delta in stream_groq(
            model=model,
            messages=messages,
            temperature=temperature,
            writer=writer,
            estimated_prompt_tokens=estimated_prompt_tokens,
            request=request,
        ):
            yield delta
    elif resolved == "gemini":
        async for delta in stream_gemini(model=model, messages=messages, temperature=temperature):
            yield delta
    elif resolved == "ollama":
        async for delta in stream_ollama(model=model, messages=messages, temperature=temperature):
            yield delta
    else:
        raise ValueError(f"Unsupported model provider: '{resolved}' for model '{model}'")
