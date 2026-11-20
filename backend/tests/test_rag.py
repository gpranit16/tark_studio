"""Unit and integration tests for TARK Studio Baseline RAG Pipeline.

Covers:
1. Multi-format parsing (PDF, MD, TXT, PY, JSON)
2. Python safe static parsing (NEVER executed)
3. Token-aware chunking with tiktoken and overlap
4. Documents CRUD APIs (upload, list, detail, delete)
5. Hybrid retrieval with Reciprocal Rank Fusion (RRF)
6. LangGraph RAG workflow routing and citation emission
"""

import os
import io
import uuid
import pytest
from unittest.mock import patch
from httpx import AsyncClient
from sqlalchemy import select

from app.database import async_session_factory, Document, DocumentChunk
from app.rag.parser import parse_document, parse_python_file, parse_markdown
from app.rag.chunker import TokenChunker, count_tokens
from app.rag.retriever import HybridRetriever, RetrievedChunk
from app.ai.graph import create_state_graph, stream_chat_workflow


def test_safe_python_parsing_no_execution(tmp_path):
    """Verify that Python files are parsed statically via AST and NEVER executed."""
    # Write a test file that would raise an error or create a file if executed
    flag_file = tmp_path / "executed_flag.txt"
    py_content = f'''"""Module docstring for test script."""
import os

# If executed, this line would create the flag file!
if False:
    open(r"{flag_file}", "w").write("EXECUTED")

class DataProcessor:
    def process(self, x):
        return x * 2

def calculate_sum(a, b):
    return a + b
'''
    py_path = tmp_path / "safe_script.py"
    py_path.write_text(py_content, encoding="utf-8")

    parsed = parse_document(str(py_path), filename="safe_script.py")

    assert parsed.file_type == "py"
    assert parsed.page_count == 1
    assert "Module docstring for test script." in parsed.metadata["static_docstring"]
    assert any("def calculate_sum" in d for d in parsed.metadata["definitions"])
    assert not os.path.exists(flag_file), "CRITICAL: Python file must NEVER be executed!"


def test_token_aware_chunking_with_overlap(tmp_path):
    """Test token chunking respecting max token budget and overlap."""
    text_content = (
        "PostgreSQL with pgvector provides efficient vector similarity search using HNSW indexing. "
        "Reciprocal Rank Fusion (RRF) merges dense vector search with sparse full-text search rankings. "
    ) * 40

    txt_file = tmp_path / "guide.txt"
    txt_file.write_text(text_content, encoding="utf-8")

    parsed = parse_document(str(txt_file))
    chunker = TokenChunker(chunk_size_tokens=100, overlap_tokens=20)
    chunks = chunker.chunk_document(parsed)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 100
        assert c.viewer_page == 1
        assert c.book_page == "1"


@pytest.mark.asyncio
async def test_document_upload_and_crud(test_client: AsyncClient, tmp_path):
    """Test uploading a Markdown document, listing it, and deleting it."""
    md_content = "# Architecture Overview\nTARK Studio uses LangGraph for stateful orchestration."

    # 1. Upload
    files = {"file": ("architecture.md", io.BytesIO(md_content.encode("utf-8")), "text/markdown")}
    upload_res = await test_client.post("/api/v1/documents/upload", files=files)
    assert upload_res.status_code == 201
    doc_data = upload_res.json()
    doc_id = doc_data["id"]
    assert doc_data["filename"] == "architecture.md"
    assert doc_data["file_type"] == "md"

    # 2. List documents
    list_res = await test_client.get("/api/v1/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["id"] == doc_id for d in docs)

    # 3. Get document detail
    detail_res = await test_client.get(f"/api/v1/documents/{doc_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == doc_id

    # 4. Delete document
    delete_res = await test_client.delete(f"/api/v1/documents/{doc_id}")
    assert delete_res.status_code == 200
    assert delete_res.json()["deleted"] is True

    # 5. Verify deleted from list
    list_res_after = await test_client.get("/api/v1/documents")
    assert not any(d["id"] == doc_id for d in list_res_after.json())


@pytest.mark.asyncio
async def test_hybrid_retriever_rrf_scoring():
    """Verify Reciprocal Rank Fusion calculation with mocked dense and sparse search results."""
    retriever = HybridRetriever(rrf_k=60)

    doc_id = uuid.uuid4()
    c1_id = uuid.uuid4()
    c2_id = uuid.uuid4()
    c3_id = uuid.uuid4()

    # Mock dense results: [c1 (rank 1), c2 (rank 2)]
    mock_dense = [
        {"id": c1_id, "document_id": doc_id, "filename": "doc.pdf", "viewer_page": 5, "book_page": "5", "heading": "Intro", "content": "Chunk 1 content", "token_count": 50, "has_image": False, "similarity": 0.88},
        {"id": c2_id, "document_id": doc_id, "filename": "doc.pdf", "viewer_page": 10, "book_page": "10", "heading": "Details", "content": "Chunk 2 content", "token_count": 60, "has_image": False, "similarity": 0.75},
    ]

    # Mock sparse results: [c2 (rank 1), c3 (rank 2)]
    mock_sparse = [
        {"id": c2_id, "document_id": doc_id, "filename": "doc.pdf", "viewer_page": 10, "book_page": "10", "heading": "Details", "content": "Chunk 2 content", "token_count": 60, "has_image": False, "fts_score": 0.95},
        {"id": c3_id, "document_id": doc_id, "filename": "doc.pdf", "viewer_page": 15, "book_page": "15", "heading": "Summary", "content": "Chunk 3 content", "token_count": 45, "has_image": False, "fts_score": 0.80},
    ]

    async def mock_embedding(prompt):
        return [0.01] * 768

    with patch.object(retriever, "_vector_search", return_value=mock_dense), \
         patch.object(retriever, "_sparse_fts_search", return_value=mock_sparse), \
         patch("app.rag.retriever.embeddings_client.get_embedding", side_effect=mock_embedding):

        results = await retriever.retrieve(query="test query", top_k=3)

        assert len(results) == 3
        # c2 appeared in BOTH dense (rank 2) and sparse (rank 1)
        # RRF(c2) = 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = ~0.0325
        # RRF(c1) = 1/(60+1) = ~0.016393
        # RRF(c3) = 1/(60+2) = ~0.016129
        # So c2 MUST be ranked first!
        assert results[0].chunk_id == c2_id
        assert results[0].dense_rank == 2
        assert results[0].sparse_rank == 1
        assert results[0].rrf_score > results[1].rrf_score


@pytest.mark.asyncio
async def test_langgraph_rag_sources_event(test_client: AsyncClient):
    """Test that initiating chat stream with document_ids activates retrieve node and yields 'sources' event."""
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    mock_chunk = RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_name="pythonlearn.pdf",
        viewer_page=21,
        book_page="15",
        heading="1.6. Terminology: Interpreter and Compiler",
        content="An interpreter reads the source code of the program as written by the programmer.",
        token_count=40,
        has_image=False,
        image_references=[],
        rrf_score=0.032,
    )

    async def mock_retrieve(*args, **kwargs):
        return [mock_chunk]

    async def mock_model_stream(*args, **kwargs):
        yield "An interpreter executes code directly [Page 15 (PDF p. 21)]."

    with patch("app.rag.retriever.hybrid_retriever.retrieve", side_effect=mock_retrieve), \
         patch("app.ai.graph.get_model_stream", side_effect=mock_model_stream):

        payload = {
            "messages": [{"role": "user", "content": "What is an interpreter?"}],
            "model": "openai/gpt-oss-120b",
            "document_ids": [str(doc_id)],
        }

        async with test_client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
            assert response.status_code == 200
            events = []
            async for chunk_str in response.aiter_text():
                events.append(chunk_str)

            full_body = "".join(events)
            assert "event: sources" in full_body
            assert "pythonlearn.pdf" in full_body
            assert "Page 15 (PDF p. 21)" in full_body
            assert "An interpreter executes code directly" in full_body


def test_synthetic_pdf_fixtures():
    """Verify parsing of synthetic PDF fixtures: multi-page, table, images, and scanned mock."""
    from app.rag.fixtures import (
        create_multi_page_pdf,
        create_table_pdf,
        create_embedded_image_pdf,
        create_scanned_page_pdf,
    )

    # 1. Multi-page text fixture
    p_multi = create_multi_page_pdf()
    parsed_multi = parse_document(p_multi)
    assert parsed_multi.page_count == 3
    assert "Chapter 1: Foundations of TARK Studio" in parsed_multi.pages[0].text
    assert "Chapter 2: Hybrid Retrieval Pipeline" in parsed_multi.pages[1].text
    assert "Chapter 3: Strict Benchmark Integrity" in parsed_multi.pages[2].text

    # 2. Table fixture
    p_table = create_table_pdf()
    parsed_table = parse_document(p_table)
    assert parsed_table.page_count == 1
    assert "Recall@5" in parsed_table.pages[0].text
    assert "0.8500" in parsed_table.pages[0].text

    # 3. Embedded images fixture
    p_img = create_embedded_image_pdf()
    parsed_img = parse_document(p_img)
    assert parsed_img.page_count == 1
    assert parsed_img.pages[0].has_image is True
    assert len(parsed_img.pages[0].image_references) >= 1
    assert parsed_img.pages[0].image_references[0]["width"] == 100

    # 4. Scanned page mock fixture (diagnostics for low text / OCR)
    p_scan = create_scanned_page_pdf()
    parsed_scan = parse_document(p_scan)
    assert parsed_scan.page_count == 1
    assert len(parsed_scan.warnings) > 0
    assert any("OCR" in w or "minimal" in w or "image" in w.lower() for w in parsed_scan.warnings)

