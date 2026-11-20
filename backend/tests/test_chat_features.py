"""Unit and integration tests for chat experience milestone features.

Covers:
- Searching conversations by title & message content
- Archiving and unarchiving conversations
- Editing user messages in PostgreSQL
- Truncating message turns for clean regeneration
- Auto-generating titles via LangGraph workflow
- Regenerate & retry with save_user_message=False preventing duplicate messages
"""

import uuid
import pytest
from unittest.mock import patch
from httpx import AsyncClient
from sqlalchemy import select

from app.database import async_session_factory, Conversation, Message
from app.ai.graph import generate_chat_title


@pytest.mark.asyncio
async def test_search_and_archive_conversations(test_client: AsyncClient):
    """Test searching by title, searching by message content, and archive/unarchive flows."""
    conv_id = uuid.uuid4()

    # Seed conversation with specific keywords
    async with async_session_factory() as session:
        conv = Conversation(
            id=conv_id,
            title="Kubernetes Deployment Guide",
            model_id="openai/gpt-oss-120b",
            provider="groq",
            is_archived=False,
        )
        session.add(conv)
        msg = Message(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            role="user",
            content="How do I configure persistent volume claims?",
            model="openai/gpt-oss-120b",
            provider="groq",
        )
        session.add(msg)
        await session.commit()

    # 1. Search by title keyword "Kubernetes"
    res = await test_client.get("/api/v1/conversations?q=Kubernetes")
    assert res.status_code == 200
    items = res.json()
    assert any(c["id"] == str(conv_id) for c in items)

    # 2. Search by message content keyword "persistent volume"
    res = await test_client.get("/api/v1/conversations?q=volume")
    assert res.status_code == 200
    items = res.json()
    assert any(c["id"] == str(conv_id) for c in items)

    # 3. Search with non-matching query
    res = await test_client.get("/api/v1/conversations?q=nonexistent_xyz_query")
    assert res.status_code == 200
    items = res.json()
    assert not any(c["id"] == str(conv_id) for c in items)

    # 4. Archive conversation
    archive_res = await test_client.post(f"/api/v1/conversations/{conv_id}/archive")
    assert archive_res.status_code == 200
    assert archive_res.json()["is_archived"] is True

    # 5. Active list should no longer include it
    active_res = await test_client.get("/api/v1/conversations?is_archived=false")
    assert not any(c["id"] == str(conv_id) for c in active_res.json())

    # 6. Archived list SHOULD include it
    archived_res = await test_client.get("/api/v1/conversations?is_archived=true")
    assert any(c["id"] == str(conv_id) for c in archived_res.json())

    # 7. Unarchive conversation
    unarchive_res = await test_client.post(f"/api/v1/conversations/{conv_id}/unarchive")
    assert unarchive_res.status_code == 200
    assert unarchive_res.json()["is_archived"] is False

    # 8. Active list includes it again
    active_res = await test_client.get("/api/v1/conversations?is_archived=false")
    assert any(c["id"] == str(conv_id) for c in active_res.json())


@pytest.mark.asyncio
async def test_edit_user_message_and_truncate(test_client: AsyncClient):
    """Test editing a previous user message and truncating subsequent messages."""
    conv_id = uuid.uuid4()
    msg1_id = uuid.uuid4()
    msg2_id = uuid.uuid4()
    msg3_id = uuid.uuid4()

    from datetime import datetime, timezone, timedelta
    t0 = datetime.now(timezone.utc)

    async with async_session_factory() as session:
        conv = Conversation(id=conv_id, title="Test Editing", model_id="openai/gpt-oss-120b", provider="groq")
        session.add(conv)
        # Turn 1: user
        session.add(Message(id=msg1_id, conversation_id=conv_id, role="user", content="Initial prompt", created_at=t0))
        # Turn 1: assistant
        session.add(Message(id=msg2_id, conversation_id=conv_id, role="assistant", content="First answer", created_at=t0 + timedelta(seconds=1)))
        # Turn 2: user
        session.add(Message(id=msg3_id, conversation_id=conv_id, role="user", content="Follow up", created_at=t0 + timedelta(seconds=2)))
        await session.commit()

    # 1. Edit msg1 content
    edit_res = await test_client.patch(
        f"/api/v1/conversations/{conv_id}/messages/{msg1_id}",
        json={"content": "Updated prompt text"},
    )
    assert edit_res.status_code == 200
    assert edit_res.json()["content"] == "Updated prompt text"

    # 2. Truncate subsequent messages after msg1
    truncate_res = await test_client.post(
        f"/api/v1/conversations/{conv_id}/truncate",
        json={"after_message_id": str(msg1_id)},
    )
    assert truncate_res.status_code == 200
    assert truncate_res.json()["deleted_count"] == 2

    # 3. Verify conversation detail only has msg1
    detail_res = await test_client.get(f"/api/v1/conversations/{conv_id}")
    assert detail_res.status_code == 200
    messages = detail_res.json()["messages"]
    assert len(messages) == 1
    assert messages[0]["id"] == str(msg1_id)
    assert messages[0]["content"] == "Updated prompt text"


@pytest.mark.asyncio
async def test_regenerate_skips_duplicate_user_message(test_client: AsyncClient):
    """Verify that calling chat/stream with save_user_message=False does NOT create duplicate user messages."""
    conv_id = uuid.uuid4()

    # Create conversation and initial user message
    async with async_session_factory() as session:
        conv = Conversation(id=conv_id, title="Test Regenerate", model_id="openai/gpt-oss-120b", provider="groq")
        session.add(conv)
        session.add(Message(id=uuid.uuid4(), conversation_id=conv_id, role="user", content="Explain Docker"))
        await session.commit()

    async def mock_stream(*args, **kwargs):
        yield "Docker is a containerization platform"

    with patch("app.ai.graph.get_model_stream", side_effect=mock_stream):
        # Regenerate: client sends save_user_message=False
        payload = {
            "conversation_id": str(conv_id),
            "messages": [{"role": "user", "content": "Explain Docker"}],
            "model": "openai/gpt-oss-120b",
            "temperature": 0.5,
            "save_user_message": False,
        }

        async with test_client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
            assert response.status_code == 200
            async for _ in response.aiter_text():
                pass

    # Verify no duplicate user messages exist in PostgreSQL
    async with async_session_factory() as session:
        stmt = select(Message).where(Message.conversation_id == conv_id)
        res = await session.execute(stmt)
        messages = res.scalars().all()

        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]

        assert len(user_messages) == 1, "There must be exactly one user message (no duplicates!)"
        assert len(assistant_messages) == 1, "There must be one newly completed assistant message"


@pytest.mark.asyncio
async def test_langgraph_title_generation(test_client: AsyncClient):
    """Verify that title generation workflow via LangGraph generates a descriptive title."""
    conv_id = uuid.uuid4()

    async with async_session_factory() as session:
        conv = Conversation(id=conv_id, title="New Chat", model_id="openai/gpt-oss-120b", provider="groq")
        session.add(conv)
        session.add(Message(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            role="user",
            content="Can you explain how transformer attention mechanisms work?",
        ))
        await session.commit()

    async def mock_title_stream(*args, **kwargs):
        yield "Transformer Attention Guide"

    with patch("app.ai.graph.get_model_stream", side_effect=mock_title_stream):
        res = await test_client.post(f"/api/v1/conversations/{conv_id}/generate-title")
        assert res.status_code == 200
        data = res.json()
        assert data["title"] == "Transformer Attention Guide"

    # Verify updated in PostgreSQL
    async with async_session_factory() as session:
        stmt = select(Conversation).where(Conversation.id == conv_id)
        conv = (await session.execute(stmt)).scalar_one_or_none()
        assert conv.title == "Transformer Attention Guide"
