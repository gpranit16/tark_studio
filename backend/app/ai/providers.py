"""Model provider abstraction for TARK Studio supporting Groq, Gemini, and Ollama."""

import logging
from typing import AsyncGenerator, List, Optional
from groq import AsyncGroq
import ollama

from app.config import settings

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
    """Instantiate AsyncGroq client using environment-configured key."""
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not configured in environment.")
    return AsyncGroq(api_key=settings.groq_api_key)


def get_ollama_client() -> ollama.AsyncClient:
    """Instantiate Ollama AsyncClient using configured base URL."""
    return ollama.AsyncClient(host=settings.ollama_base_url)


async def stream_groq(
    model: str,
    messages: List[dict],
    temperature: float,
) -> AsyncGenerator[str, None]:
    """Stream token chunks from Groq."""
    client = get_groq_client()
    formatted = [{"role": m["role"], "content": m["content"]} for m in messages]

    response_stream = await client.chat.completions.create(
        model=model,
        messages=formatted,
        temperature=temperature,
        stream=True,
    )

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
) -> AsyncGenerator[str, None]:
    """Unified dispatcher streaming token chunks from the selected provider."""
    resolved = provider or resolve_provider(model)

    if resolved == "groq":
        async for delta in stream_groq(model=model, messages=messages, temperature=temperature):
            yield delta
    elif resolved == "gemini":
        async for delta in stream_gemini(model=model, messages=messages, temperature=temperature):
            yield delta
    elif resolved == "ollama":
        async for delta in stream_ollama(model=model, messages=messages, temperature=temperature):
            yield delta
    else:
        raise ValueError(f"Unsupported model provider: '{resolved}' for model '{model}'")
