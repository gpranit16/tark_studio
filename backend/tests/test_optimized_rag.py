"""TARK Studio — Verification Tests for Optimized RAG Ingestion & Pipeline."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
import fitz

from app.rag.embeddings import (
    CloudflareEmbeddings,
    EmbeddingClientProxy,
    _query_cache,
    EMBEDDING_DIM,
)
from app.rag.parser import parse_pdf, _extract_visible_urls
from app.config import settings


@pytest.mark.asyncio
async def test_cloudflare_concurrent_batch_execution():
    """Verify concurrent batch execution handles multiple slices in exact order."""
    provider = CloudflareEmbeddings(
        account_id="test_acc",
        api_token="test_tok",
        batch_size=4,
    )
    
    texts = [f"Text chunk {i}" for i in range(10)]  # 3 batches: 4, 4, 2
    
    def side_effect(url, headers, json):
        input_texts = json.get("input", [])
        mock_res = MagicMock(spec=httpx.Response)
        mock_res.status_code = 200
        # Return unique embeddings based on chunk text to verify ordering
        mock_res.json.return_value = {
            "success": True,
            "result": {
                "data": [[float(i.split()[-1])] * 768 for i in input_texts]
            }
        }
        return mock_res

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(side_effect=side_effect)
    provider._client = mock_client

    results = await provider.get_embeddings_batch(texts, is_query=False)
    
    assert len(results) == 10
    for idx, emb in enumerate(results):
        assert emb[0] == float(idx)
        assert len(emb) == 768


@pytest.mark.asyncio
async def test_query_embedding_lru_cache():
    """Verify that repeated queries hit the in-memory LRU cache with 0 network calls."""
    provider = CloudflareEmbeddings(
        account_id="test_acc",
        api_token="test_tok",
    )
    
    fake_vector = [0.42] * 768
    mock_res = MagicMock(spec=httpx.Response)
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "success": True,
        "result": {"data": [fake_vector]}
    }
    
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=mock_res)
    provider._client = mock_client

    query = "Unique search query for cache test"
    
    # First call: hits provider and caches
    emb1 = await provider.get_embedding(query, is_query=True)
    assert emb1 == fake_vector
    assert mock_client.post.call_count == 1

    # Second call: served directly from cache (call_count remains 1)
    emb2 = await provider.get_embedding(query, is_query=True)
    assert emb2 == fake_vector
    assert mock_client.post.call_count == 1


def test_proxy_dynamic_routing_production_vs_development():
    """Verify that proxy routes to Cloudflare in production with creds, and Ollama in local dev."""
    proxy = EmbeddingClientProxy()

    # 1. Local Development (app_env = development, default provider = ollama)
    with patch.object(settings, "app_env", "development"):
        with patch.object(settings, "embedding_provider", "ollama"):
            provider = proxy._resolve_provider()
            assert provider.model_name == "nomic-embed-text"

    # 2. Production (app_env = production, Cloudflare credentials provided)
    with patch.object(settings, "app_env", "production"):
        with patch.object(settings, "cloudflare_account_id", "mock_cf_id"):
            with patch.object(settings, "cloudflare_api_token", "mock_cf_token"):
                with patch.object(settings, "embedding_provider", "ollama"):  # even if unset or default
                    provider = proxy._resolve_provider()
                    assert provider.model_name == "@cf/google/embeddinggemma-2"


def test_pdf_plain_text_url_and_heading_extraction(tmp_path):
    """Verify that PyMuPDF parser extracts visible URLs and title-case section headings."""
    pdf_path = str(tmp_path / "test_doc.pdf")
    doc = fitz.open()
    page = doc.new_page()
    
    # Insert text with heading, visible URL, and body
    sample_text = (
        "Engineering Experience\n"
        "Lead Architect at Global Corp.\n"
        "Portfolio: https://portfolio.example.org/projects?view=summary\n"
        "Contact: engineer@example.com\n"
    )
    page.insert_text((50, 72), sample_text, fontsize=11)
    doc.save(pdf_path)
    doc.close()

    parsed = parse_pdf(pdf_path, "test_doc.pdf")
    assert parsed.page_count == 1
    p0 = parsed.pages[0]

    # Heading should be extracted
    assert p0.heading == "Engineering Experience"

    # URL should be captured in links
    uris = [l["uri"] for l in p0.links]
    assert any("https://portfolio.example.org/projects?view=summary" in u for u in uris)

    # Clean text should contain the extracted link references
    assert "Hyperlinks & References" in p0.text
    assert "https://portfolio.example.org" in p0.text
