"""Integration tests for token-by-token streaming endpoint via LangGraph."""

import json
import pytest
from unittest.mock import patch
from httpx import AsyncClient
from app.config import settings


@pytest.mark.asyncio
async def test_chat_stream_endpoint_flow(test_client: AsyncClient):
    """Verify end-to-end SSE token-by-token streaming via FastAPI and LangGraph."""
    async def mock_stream(*args, **kwargs):
        yield "Testing "
        yield "LangGraph "
        yield "streaming"

    with patch("app.ai.graph.get_model_stream", side_effect=mock_stream):
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "openai/gpt-oss-120b",
            "temperature": 0.5,
        }

        events_received = []
        tokens_received = []

        async with test_client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event_block, buffer = buffer.split("\n\n", 1)
                    lines = event_block.strip().split("\n")
                    event_type = ""
                    data_str = ""
                    for line in lines:
                        if line.startswith("event: "):
                            event_type = line.split("event: ")[1].strip()
                        elif line.startswith("data: "):
                            data_str = line.split("data: ")[1].strip()

                    if event_type:
                        events_received.append(event_type)
                    if event_type == "token" and data_str:
                        data = json.loads(data_str)
                        if "delta" in data:
                            tokens_received.append(data["delta"])

        assert "meta" in events_received, "Stream must emit meta event"
        assert "token" in events_received, "Stream must emit token events"
        assert "done" in events_received, "Stream must emit done event"
        assert "".join(tokens_received) == "Testing LangGraph streaming"


@pytest.mark.asyncio
async def test_live_groq_stream(test_client: AsyncClient):
    """Verify live SSE token-by-token streaming from Groq via FastAPI (if network available)."""
    if not settings.groq_api_key:
        pytest.skip("GROQ_API_KEY is not set.")

    payload = {
        "messages": [
            {"role": "user", "content": "Respond with the word 'HELLO' and nothing else."}
        ],
        "model": "openai/gpt-oss-120b",
        "temperature": 0.2,
    }

    events_received = []
    tokens_received = []

    async with test_client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                event_block, buffer = buffer.split("\n\n", 1)
                lines = event_block.strip().split("\n")
                event_type = ""
                data_str = ""
                for line in lines:
                    if line.startswith("event: "):
                        event_type = line.split("event: ")[1].strip()
                    elif line.startswith("data: "):
                        data_str = line.split("data: ")[1].strip()

                if event_type:
                    events_received.append(event_type)
                if event_type == "error" and data_str:
                    err_obj = json.loads(data_str)
                    err_text = str(err_obj.get("error", ""))
                    if "not allowed by policy" in err_text or "ConnectError" in err_text:
                        pytest.skip(f"Live network access blocked by environment policy: {err_text}")
                if event_type == "token" and data_str:
                    data = json.loads(data_str)
                    if "delta" in data:
                        tokens_received.append(data["delta"])

    assert "meta" in events_received, "Stream must emit meta event"
    assert "token" in events_received, "Stream must emit token events"
    assert "done" in events_received, "Stream must emit done event"
    accumulated_text = "".join(tokens_received).strip()
    assert len(accumulated_text) > 0, "Accumulated text must not be empty"
    assert "HELLO" in accumulated_text.upper()
