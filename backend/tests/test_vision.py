"""Unit tests for Unified Vision OCR Client (Gemini & Cloudflare)."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.rag.vision import UnifiedVisionOCRClient, CloudflareVisionClient


@pytest.mark.asyncio
async def test_vision_unconfigured_returns_none():
    """Verify that when all vision credentials are missing, transcribe returns None without error."""
    client = UnifiedVisionOCRClient()
    client.gemini_api_key = None
    client.cloudflare_account_id = None
    client.cloudflare_api_token = None

    with patch("shutil.which", return_value=None):
        result = await client.transcribe_image_bytes(b"dummy_bytes")
        assert result is None


@pytest.mark.asyncio
async def test_vision_gemini_success_mocked():
    """Verify successful transcription extraction from Google Gemini Flash Vision."""
    client = UnifiedVisionOCRClient()
    client.gemini_api_key = "test_gemini_key"
    client.cloudflare_account_id = None
    client.cloudflare_api_token = None

    mock_resp = MagicMock()
    mock_resp.text = "Role: AI Engineer Intern\nStipend: Rs. 35,000/month"

    mock_aio = MagicMock()
    mock_aio.models.generate_content = AsyncMock(return_value=mock_resp)
    mock_client = MagicMock()
    mock_client.aio = mock_aio

    with patch("google.genai.Client", return_value=mock_client):
        res = await client.transcribe_image_bytes(b"dummy_bytes")
        assert res is not None
        assert "AI Engineer Intern" in res
        assert "35,000" in res


@pytest.mark.asyncio
async def test_vision_cloudflare_transcribe_success_mocked():
    """Verify successful transcription extraction from Cloudflare response payload."""
    client = CloudflareVisionClient()
    client.gemini_api_key = None
    client.cloudflare_account_id = "test_acc"
    client.cloudflare_api_token = "test_tok"

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
async def test_vision_cloudflare_transcribe_429_retry_handling():
    """Verify that HTTP 429 triggers retry with Retry-After inspection."""
    client = CloudflareVisionClient()
    client.gemini_api_key = None
    client.cloudflare_account_id = "test_acc"
    client.cloudflare_api_token = "test_tok"
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
