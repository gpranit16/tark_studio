"""Unit tests for LangGraph AI workflow orchestration and provider resolution."""

import pytest
from unittest.mock import patch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.ai.state import ChatState
from app.ai.graph import (
    chat_graph,
    build_chat_graph,
    get_checkpointer,
    set_test_checkpointer,
    close_checkpointer,
)
from app.ai.providers import resolve_provider, SUPPORTED_MODELS


def test_resolve_provider():
    """Verify provider resolution across Groq, Gemini, and Ollama models."""
    assert resolve_provider("openai/gpt-oss-120b") == "groq"
    assert resolve_provider("openai/gpt-oss-20b") == "groq"
    assert resolve_provider("qwen/qwen3.8-27b") == "groq"
    assert resolve_provider("llama-3.3-70b-versatile") == "groq"
    assert resolve_provider("gemini-2.5-flash") == "gemini"
    assert resolve_provider("gemini-2.5-pro") == "gemini"
    assert resolve_provider("llama3.2") == "ollama"
    assert resolve_provider("qwen2.5") == "ollama"


def test_supported_models_structure():
    """Verify SUPPORTED_MODELS contains all required fields."""
    assert len(SUPPORTED_MODELS) >= 6
    for model in SUPPORTED_MODELS:
        assert "id" in model
        assert "name" in model
        assert "provider" in model
        assert model["provider"] in ["groq", "gemini", "ollama"]


@pytest.mark.asyncio
async def test_langgraph_chat_workflow_mocked():
    """Verify that LangGraph executes chat_node and streams tokens via StreamWriter."""
    async def mock_stream(*args, **kwargs):
        yield "Hello"
        yield " from"
        yield " LangGraph!"

    with patch("app.ai.graph.get_model_stream", side_effect=mock_stream):
        initial_state: ChatState = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "openai/gpt-oss-120b",
            "temperature": 0.5,
            "provider": "groq",
            "response": "",
            "error": None,
        }

        emitted_events = []
        config = {"configurable": {"thread_id": "test-thread-1"}}
        async for chunk in chat_graph.astream(initial_state, stream_mode="custom", config=config):
            emitted_events.append(chunk)

        event_types = [e["type"] for e in emitted_events]
        assert "meta" in event_types, "Must emit meta event"
        assert "token" in event_types, "Must emit token events"
        assert "done" in event_types, "Must emit done event"

        # Check token payloads
        tokens = [e["data"]["delta"] for e in emitted_events if e["type"] == "token"]
        assert "".join(tokens) == "Hello from LangGraph!"


@pytest.mark.asyncio
async def test_checkpointer_raises_when_postgres_fails():
    """Verify that get_checkpointer raises a clear RuntimeError when Postgres fails, forbidding silent fallback."""
    await close_checkpointer()

    with patch("app.ai.graph.AsyncConnectionPool.open", side_effect=ConnectionRefusedError("DB offline")):
        with pytest.raises(RuntimeError) as exc_info:
            await get_checkpointer()

        assert "PostgreSQL checkpointer is unavailable" in str(exc_info.value)
        assert "Silent fallback to in-memory storage is disabled" in str(exc_info.value)


@pytest.mark.asyncio
async def test_explicit_test_checkpointer_allowed():
    """Verify that MemorySaver is permitted only when explicitly configured for tests."""
    await close_checkpointer()
    custom_memory_saver = MemorySaver()
    set_test_checkpointer(custom_memory_saver)

    try:
        checkpointer = await get_checkpointer()
        assert checkpointer is custom_memory_saver
    finally:
        set_test_checkpointer(None)
        await close_checkpointer()
