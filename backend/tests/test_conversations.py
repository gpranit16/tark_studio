"""Integration tests for persistent conversation CRUD and message loading."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_conversation_lifecycle(test_client: AsyncClient):
    """Verify create, list, load, rename, and delete conversation APIs."""
    # 1. Create a new conversation
    create_res = await test_client.post(
        "/api/v1/conversations",
        json={"title": "Test Thread", "model_id": "openai/gpt-oss-120b", "provider": "groq"},
    )
    assert create_res.status_code == 201
    created_data = create_res.json()
    conv_id = created_data["id"]
    assert created_data["title"] == "Test Thread"
    assert created_data["messages"] == []

    # 2. List conversations (should include newly created conversation)
    list_res = await test_client.get("/api/v1/conversations")
    assert list_res.status_code == 200
    conversations = list_res.json()
    assert any(c["id"] == conv_id for c in conversations)

    # 3. Load specific conversation
    get_res = await test_client.get(f"/api/v1/conversations/{conv_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == conv_id

    # 4. Rename conversation
    patch_res = await test_client.patch(
        f"/api/v1/conversations/{conv_id}",
        json={"title": "Renamed Session"},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["title"] == "Renamed Session"

    # 5. Delete conversation
    del_res = await test_client.delete(f"/api/v1/conversations/{conv_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # 6. Verify 404 after deletion
    not_found_res = await test_client.get(f"/api/v1/conversations/{conv_id}")
    assert not_found_res.status_code == 404
