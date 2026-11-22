"""Integration tests for Stop Generation / client cancellation behavior."""

import uuid
import pytest
from unittest.mock import patch
from httpx import AsyncClient
from sqlalchemy import select

from app.database import async_session_factory, Message, Conversation


@pytest.mark.asyncio
async def test_stream_cancellation_does_not_save_incomplete_assistant_response(test_client: AsyncClient):
    """Verify that when a client stops/aborts stream, partial assistant messages are NOT saved in PostgreSQL."""
    conv_id = uuid.uuid4()

    async def mock_slow_stream(*args, **kwargs):
        yield "First token"
        yield " second token"
        yield " third token should never be reached"

    call_count = 0
    async def mock_is_disconnected(self):
        nonlocal call_count
        call_count += 1
        # After first check, simulate client disconnect
        return call_count > 1

    with patch("app.ai.graph.get_model_stream", side_effect=mock_slow_stream), \
         patch("starlette.requests.Request.is_disconnected", new=mock_is_disconnected):
        payload = {
            "conversation_id": str(conv_id),
            "messages": [{"role": "user", "content": "Tell me a long story"}],
            "model": "openai/gpt-oss-120b",
            "temperature": 0.5,
        }

        # Client opens stream and consumes whatever server sends until server halts
        async with test_client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
            assert response.status_code == 200
            chunks = []
            async for chunk in response.aiter_text():
                chunks.append(chunk)

        # Allow cancellation to propagate to background tasks before assertions
        await asyncio.sleep(0.1)

        # Verify PostgreSQL database state:
        # User message was saved concurrently
        # But INCOMPLETE assistant response must NOT be saved!
        async with async_session_factory() as session:
            stmt = select(Message).where(Message.conversation_id == conv_id)
            res = await session.execute(stmt)
            messages = res.scalars().all()

            assistant_messages = [m for m in messages if m.role == "assistant"]
            assert len(assistant_messages) == 0, "Incomplete assistant response must NOT be saved in DB upon cancellation!"


@pytest.mark.asyncio
async def test_completed_stream_saves_assistant_response(test_client: AsyncClient):
    """Verify that fully completed generation DOES save the assistant response to PostgreSQL."""
    conv_id = uuid.uuid4()

    async def mock_full_stream(*args, **kwargs):
        yield "Full"
        yield " response"

    with patch("app.ai.graph.get_model_stream", side_effect=mock_full_stream):
        payload = {
            "conversation_id": str(conv_id),
            "messages": [{"role": "user", "content": "Quick question"}],
            "model": "openai/gpt-oss-120b",
            "temperature": 0.5,
        }

        # Client reads full stream to completion
        async with test_client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
            assert response.status_code == 200
            full_body = ""
            async for chunk in response.aiter_text():
                full_body += chunk

            assert "Full" in full_body
            assert "response" in full_body

        # Verify assistant message IS persisted in PostgreSQL
        async with async_session_factory() as session:
            stmt = select(Message).where(Message.conversation_id == conv_id)
            res = await session.execute(stmt)
            messages = res.scalars().all()

            assistant_messages = [m for m in messages if m.role == "assistant"]
            assert len(assistant_messages) == 1, "Completed assistant response MUST be saved in DB!"
            assert assistant_messages[0].content == "Full response"
