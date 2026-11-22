"""TARK Studio — Tests for Multi-Provider Embeddings Pipeline."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.rag.embeddings import (
    BaseEmbeddingProvider,
    OllamaEmbeddings,
    CloudflareEmbeddings,
    EmbeddingClientProxy,
    EMBEDDING_DIM,
)
from app.config import settings


@pytest.mark.asyncio
async def test_ollama_provider_defaults():
    """Ensure Ollama provider preserves existing local defaults."""
    provider = OllamaEmbeddings()
    assert provider.dimension == 768
    assert provider.model_name == "nomic-embed-text"


@pytest.mark.asyncio
async def test_cloudflare_provider_dimensions_and_model():
    """Ensure Cloudflare provider targets @cf/google/embeddinggemma-2 with 768 dimensions."""
    provider = CloudflareEmbeddings(
        account_id="mock_acc_id",
        api_token="mock_token",
        model="@cf/google/embeddinggemma-2",
    )
    assert provider.dimension == 768
    assert provider.model_name == "@cf/google/embeddinggemma-2"


def test_cloudflare_input_formatting():
    """Ensure task instruction prefix is applied to search queries."""
    provider = CloudflareEmbeddings(account_id="acc", api_token="tok")
    
    # Query must have task instruction prefix
    q_fmt = provider._format_input("What is TARK Studio?", is_query=True)
    assert q_fmt == "task: search result | query: What is TARK Studio?"

    # Already formatted query is not double-prefixed
    q_pre = provider._format_input("task: search result | query: What is TARK Studio?", is_query=True)
    assert q_pre == "task: search result | query: What is TARK Studio?"

    # Passage / document chunk is left as clean content
    doc_fmt = provider._format_input("TARK Studio is a local-first workspace.", is_query=False)
    assert doc_fmt == "TARK Studio is a local-first workspace."


@pytest.mark.asyncio
async def test_cloudflare_embeddings_success_dict_format():
    """Ensure Cloudflare parser handles [{'embedding': [...]}] response schema."""
    fake_vector = [0.1] * 768
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "result": {
            "data": [{"embedding": fake_vector}],
            "shape": [1, 768],
        },
    }

    provider = CloudflareEmbeddings(account_id="acc123", api_token="tok456")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await provider.get_embedding("Test search", is_query=True)

        assert len(result) == 768
        assert result == fake_vector
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "acc123/ai/run/@cf/google/embeddinggemma-2" in call_url
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer tok456"


@pytest.mark.asyncio
async def test_cloudflare_embeddings_success_list_format():
    """Ensure Cloudflare parser handles [[...]] float array response schema."""
    fake_vector = [-0.05] * 768
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "result": {
            "data": [fake_vector],
            "shape": [1, 768],
        },
    }

    provider = CloudflareEmbeddings(account_id="acc123", api_token="tok456")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        results = await provider.get_embeddings_batch(["Sample document passage"], is_query=False)

        assert len(results) == 1
        assert len(results[0]) == 768
        assert results[0] == fake_vector


@pytest.mark.asyncio
async def test_cloudflare_missing_credentials_raises_error():
    """Ensure missing Cloudflare credentials raise clear configuration error."""
    provider = CloudflareEmbeddings(account_id="", api_token="")
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        await provider.get_embeddings_batch(["Some text"])


@pytest.mark.asyncio
async def test_proxy_dynamic_provider_selection():
    """Verify EmbeddingClientProxy switches providers based on settings without breaking callers."""
    proxy = EmbeddingClientProxy()

    with patch.object(settings, "embedding_provider", "ollama"):
        assert isinstance(proxy._resolve_provider(), OllamaEmbeddings)
        assert proxy.model_name == "nomic-embed-text"

    with patch.object(settings, "embedding_provider", "cloudflare"):
        # Reset proxy cache for test
        proxy._provider_cache.clear()
        assert isinstance(proxy._resolve_provider(), CloudflareEmbeddings)
        assert proxy.model_name == "@cf/google/embeddinggemma-2"
