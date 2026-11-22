"""Unit and Regression Tests for Evidence Retrieval, Grounding, and Source-of-Truth Hierarchy."""
import pytest
import uuid
from app.rag.query_resolver import (
    resolve_conversational_query,
    extract_entities_from_history,
    ResolvedQuery,
)
from app.ai.graph import evaluate_is_off_topic
from app.rag.retriever import RetrievedChunk


def make_test_chunk(
    doc_id: uuid.UUID,
    content: str,
    dense: float = 0.45,
    sparse: float = 0.0,
    ce: float = -9.0,
    page: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_name="resume.pdf",
        viewer_page=page,
        book_page=str(page),
        heading="Profile",
        content=content,
        token_count=len(content.split()),
        has_image=False,
        image_references=[],
        dense_rank=1,
        sparse_rank=1,
        rrf_score=0.03,
        dense_score=dense,
        sparse_score=sparse,
        rerank_score=ce,
        rerank_rank=1,
    )


def test_initial_query_standalone():
    """Initial user queries with no prior assistant dialogue must remain untouched."""
    messages = [{"role": "user", "content": "whose resume is this ??"}]
    resolved = resolve_conversational_query("whose resume is this ??", messages)
    assert resolved.search_query == "whose resume is this ??"
    assert not resolved.is_followup
    assert not resolved.is_correction
    assert resolved.entity is None


def test_off_topic_query_not_falsely_resolved():
    """Independent off-topic questions must never be contaminated with entities from history."""
    messages = [
        {"role": "user", "content": "whose resume is this ??"},
        {"role": "assistant", "content": "The resume belongs to Pranit Kumar."},
        {"role": "user", "content": "What is the capital of France and what is its population?"},
    ]
    resolved = resolve_conversational_query(
        "What is the capital of France and what is its population?",
        messages,
    )
    assert not resolved.is_followup
    assert resolved.search_query == "What is the capital of France and what is its population?"


def test_conversational_pronoun_resolution():
    """Follow-up questions with pronouns must resolve the entity from prior dialogue."""
    messages = [
        {"role": "user", "content": "whose resume is this ??"},
        {"role": "assistant", "content": "The resume belongs to **Pranit Kumar**."},
    ]
    resolved = resolve_conversational_query("what is his linkedin account ?", messages)
    assert resolved.is_followup
    assert resolved.entity == "Pranit Kumar"
    assert "Pranit Kumar" in resolved.search_query
    assert "linkedin" in resolved.search_query.lower()

    resolved_loc = resolve_conversational_query("where is he from ?", messages)
    assert resolved_loc.is_followup
    assert resolved_loc.entity == "Pranit Kumar"
    assert "location" in resolved_loc.search_query.lower()


def test_conflict_correction_resolution():
    """User corrections must signal re-verification from the original source document.
    The search query must NOT hardcode location names — it must be generic.
    """
    messages = [
        {"role": "user", "content": "where is he from ?"},
        {"role": "assistant", "content": "He is from Bhopal, Madhya Pradesh."},
    ]
    resolved = resolve_conversational_query(
        "then why you told madhya pradesh >",
        messages,
    )
    assert resolved.is_followup
    assert resolved.is_correction
    # Generic re-verification signal — no hardcoded location names
    assert "verify" in resolved.search_query or "confirm" in resolved.search_query or "source" in resolved.search_query
    # The user's original query terms must be preserved
    assert "madhya pradesh" in resolved.search_query.lower()


def test_off_topic_consensus_evaluation():
    """Multi-channel consensus must flag genuine off-topic queries as off-topic."""
    doc_id = uuid.uuid4()
    # Unrelated candidates with low dense similarity and floor cross-encoder scores
    chunks = [
        make_test_chunk(doc_id, "Candidate experience in Node.js and React", dense=0.35, sparse=0.0, ce=-11.25),
        make_test_chunk(doc_id, "Education at BMS Institute of Technology", dense=0.38, sparse=0.0, ce=-11.30),
    ]
    is_off = evaluate_is_off_topic(
        query="What is the capital of France?",
        chunks=chunks,
        candidate_pool_size=len(chunks),
        reranker_type="onnx",
    )
    assert is_off is True


def test_on_topic_evaluation_acceptance():
    """Multi-channel consensus must accept on-topic queries with strong cross-encoder score."""
    doc_id = uuid.uuid4()
    chunks = [
        make_test_chunk(doc_id, "Bengaluru, India | +91-9508746855", dense=0.58, sparse=0.05, ce=-2.30),
    ]
    is_off = evaluate_is_off_topic(
        query="Pranit Kumar where is he from",
        chunks=chunks,
        candidate_pool_size=len(chunks),
        reranker_type="onnx",
    )
    assert is_off is False
