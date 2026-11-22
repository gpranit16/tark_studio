"""Unit tests for Cloudflare Workers AI Vision OCR Client."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.rag.vision import CloudflareVisionClient


@pytest.mark.asyncio
async def test_vision_unconfigured_returns_none():
    """Verify that when Cloudflare credentials are missing, transcribe returns None without error."""
    client = CloudflareVisionClient()
    client.account_id = None
    client.api_token = None

    result = await client.transcribe_image_bytes(b"dummy_bytes")
    assert result is None


@pytest.mark.asyncio
async def test_vision_transcribe_success_mocked():
    """Verify successful transcription extraction from Cloudflare response payload."""
    client = CloudflareVisionClient()
    client.account_id = "test_acc"
    client.api_token = "test_tok"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "result": {
            "response": "Offer Letter: Software Engineer Intern\nCompensation: 25,000 INR/month"
        }
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await client.transcribe_image_bytes(b"dummy_bytes")
        assert res is not None
        assert "Software Engineer Intern" in res
        assert "25,000 INR/month" in res


@pytest.mark.asyncio
async def test_vision_transcribe_429_retry_handling():
    """Verify that HTTP 429 triggers retry with Retry-After inspection."""
    client = CloudflareVisionClient()
    client.account_id = "test_acc"
    client.api_token = "test_tok"
    client.max_retries = 1

    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.headers = {"retry-after": "0.01"}

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {"result": {"response": "Transcribed successfully on retry"}}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=[mock_429, mock_200]):
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            res = await client.transcribe_image_bytes(b"dummy_bytes")
            assert res == "Transcribed successfully on retry"
            assert mock_sleep.call_count >= 1
