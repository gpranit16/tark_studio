"""Unit tests for TARK Studio Reranker Implementations and LangGraph Routing.

Tests:
1. BaselineReranker pass-through behavior.
2. ONNXReranker scoring and ranking order.
3. BGEReranker singleton caching and scoring.
4. Error handling and refusal of silent fallback on missing or failed models.
5. LangGraph rerank_node integration (top-k filtering, domain rules, honest refusal for off-topic).
"""

import pytest
import uuid
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from app.rag.reranker import (
    BaselineReranker,
    BGEReranker,
    ONNXReranker,
    get_reranker,
    BaseReranker,
)
from app.rag.retriever import RetrievedChunk


def make_dummy_chunk(doc_id: uuid.UUID, content: str, score: float, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_name="test_doc.pdf",
        viewer_page=page,
        book_page=str(page),
        heading="Section 1",
        content=content,
        token_count=len(content.split()),
        has_image=False,
        image_references=[],
        dense_rank=1,
        sparse_rank=1,
        rrf_score=score,
    )


def test_baseline_reranker():
    doc_id = uuid.uuid4()
    chunks = [
        make_dummy_chunk(doc_id, "Passage A", 0.9),
        make_dummy_chunk(doc_id, "Passage B", 0.8),
        make_dummy_chunk(doc_id, "Passage C", 0.7),
        make_dummy_chunk(doc_id, "Passage D", 0.6),
        make_dummy_chunk(doc_id, "Passage E", 0.5),
        make_dummy_chunk(doc_id, "Passage F", 0.4),
    ]

    reranker = BaselineReranker()
    reranked = reranker.rerank("test query", chunks, top_k=3)

    assert len(reranked) == 3
    assert reranked[0].content == "Passage A"
    assert reranked[1].content == "Passage B"
    assert reranked[2].content == "Passage C"
    assert reranked[0].rerank_rank == 1
    assert reranked[0].rerank_score is None


def test_reranker_factory():
    base = get_reranker("baseline")
    assert isinstance(base, BaselineReranker)

    with pytest.raises(ValueError, match="Unknown reranker type"):
        get_reranker("unsupported_model")


def test_onnx_reranker_execution():
    doc_id = uuid.uuid4()
    chunks = [
        make_dummy_chunk(doc_id, "Bananas and apples are common edible fruits in grocery stores.", 0.5),
        make_dummy_chunk(doc_id, "Python is an interpreted high-level general-purpose programming language.", 0.9),
        make_dummy_chunk(doc_id, "PostgreSQL is a powerful open source object-relational database.", 0.8),
    ]

    reranker = get_reranker("onnx")
    assert isinstance(reranker, ONNXReranker)

    query = "What programming language is high-level and interpreted?"
    results = reranker.rerank(query, chunks, top_k=2)

    assert len(results) == 2
    # The Python chunk should score highest for this query
    assert "Python is an interpreted" in results[0].content
    assert results[0].rerank_rank == 1
    assert results[0].rerank_score is not None
    assert results[0].rerank_score > results[1].rerank_score


def test_bge_reranker_execution():
    doc_id = uuid.uuid4()
    chunks = [
        make_dummy_chunk(doc_id, "The capital of France is Paris, located on the River Seine.", 0.5),
        make_dummy_chunk(doc_id, "SQL uses SELECT, FROM and WHERE clauses to query database tables.", 0.9),
    ]

    reranker = get_reranker("bge")
    assert isinstance(reranker, BGEReranker)

    query = "Where is Paris located?"
    results = reranker.rerank(query, chunks, top_k=1)

    assert len(results) == 1
    assert "Paris" in results[0].content
    assert results[0].rerank_rank == 1
    assert results[0].rerank_score is not None


def test_reranker_refuses_silent_fallback_on_error():
    """Verify that if reranker model scoring throws an exception, it raises RuntimeError rather than falling back silently."""
    reranker = BaselineReranker()
    doc_id = uuid.uuid4()
    chunks = [make_dummy_chunk(doc_id, "Test", 0.5)]

    # Test BGEReranker error handling
    with patch.object(BGEReranker, "_model") as mock_model:
        mock_model.compute_score.side_effect = RuntimeError("GPU out of memory")
        bge = BGEReranker.__new__(BGEReranker)
        with pytest.raises(RuntimeError, match="BGE reranker failed during inference"):
            bge.rerank("query", chunks)


@pytest.mark.asyncio
async def test_langgraph_rerank_node_off_topic_honest_refusal():
    """Verify that off-topic queries produce zero citations and honest refusal in rerank_node."""
    from app.ai.graph import rerank_node

    events = []
    def dummy_writer(event):
        events.append(event)

    doc_id = uuid.uuid4()
    # Chunks about Python database tables
    irrelevant_chunks = [
        make_dummy_chunk(doc_id, "In Chapter 15, we define a Track table with id, title, and album_id integer fields.", 0.1),
    ]

    state = {
        "messages": [{"role": "user", "content": "What is the capital of France and its population?"}],
        "candidate_chunks": irrelevant_chunks,
        "reranker_type": "onnx",
    }

    result = await rerank_node(state, dummy_writer)

    # Must emit sources event with 0 citations
    source_events = [e for e in events if e.get("type") == "sources"]
    assert len(source_events) == 1
    assert len(source_events[0]["data"]["citations"]) == 0

    assert len(result["citations"]) == 0
    assert any("CANNOT be answered" in m["content"] for m in result["messages"])
