"""LangGraph AI Orchestration for TARK Studio chat workflows with AsyncPostgresSaver."""

import sys
import re
import json
import logging
import uuid
import asyncio
import time

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
from typing import AsyncGenerator, List, Optional, Any, Dict
from fastapi import Request
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.ai.state import ChatState, TitleState
from app.ai.providers import get_model_stream, resolve_provider
from app.ai.rate_limiter import groq_rate_limiter
from app.ai.memory import (
    partition_messages,
    should_update_summary,
    generate_rolling_summary,
    background_summary_manager,
    count_messages_tokens,
    count_tokens,
)

logger = logging.getLogger(__name__)

# Checkpointer singletons
_checkpointer_pool: Optional[AsyncConnectionPool] = None
_postgres_checkpointer: Optional[AsyncPostgresSaver] = None
_active_compiled_graph = None
_test_checkpointer: Optional[BaseCheckpointSaver] = None


def set_test_checkpointer(checkpointer: Optional[BaseCheckpointSaver]):
    """Explicitly configure an in-memory or mock checkpointer for tests only.
    
    Silent in-memory fallback is disallowed in runtime environments.
    """
    global _test_checkpointer, _active_compiled_graph
    _test_checkpointer = checkpointer
    _active_compiled_graph = None


def get_postgres_connection_string() -> str:
    """Resolve standard psycopg connection string for checkpointer."""
    if settings.database_url:
        url = settings.database_url
        if url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
        return url
    return (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}@"
        f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


async def get_checkpointer() -> BaseCheckpointSaver:
    """Return persistent AsyncPostgresSaver backed by AsyncConnectionPool.
    
    Raises RuntimeError if PostgreSQL is unavailable.
    Silent fallback to MemorySaver is forbidden; MemorySaver is only permitted
    when explicitly configured via set_test_checkpointer().
    """
    global _checkpointer_pool, _postgres_checkpointer, _test_checkpointer

    # Explicit test checkpointer permitted only if explicitly registered
    if _test_checkpointer is not None:
        return _test_checkpointer

    if _postgres_checkpointer is not None and _checkpointer_pool is not None and not _checkpointer_pool.closed:
        return _postgres_checkpointer

    try:
        conn_string = get_postgres_connection_string()
        if _checkpointer_pool is None or _checkpointer_pool.closed:
            _checkpointer_pool = AsyncConnectionPool(
                conninfo=conn_string,
                max_size=10,
                min_size=1,
                max_idle=60.0,
                check=AsyncConnectionPool.check_connection,
                open=False,
            )
            await _checkpointer_pool.open()

        _postgres_checkpointer = AsyncPostgresSaver(conn=_checkpointer_pool)
        await _postgres_checkpointer.setup()
        logger.info("LangGraph AsyncPostgresSaver initialized successfully with AsyncConnectionPool.")
        return _postgres_checkpointer
    except Exception as exc:
        logger.error(f"PostgreSQL checkpointer initialization failed: {exc}")
        if _checkpointer_pool is not None:
            try:
                await _checkpointer_pool.close()
            except Exception:
                pass
            _checkpointer_pool = None
        _postgres_checkpointer = None
        raise RuntimeError(
            f"PostgreSQL checkpointer is unavailable: {exc}. "
            "PostgreSQL checkpointing is required for TARK Studio session state. "
            "Silent fallback to in-memory storage is disabled."
        )


async def close_checkpointer():
    """Dispose of checkpointer connection pool during shutdown."""
    global _checkpointer_pool, _postgres_checkpointer, _active_compiled_graph, _test_checkpointer
    _active_compiled_graph = None
    _test_checkpointer = None
    _postgres_checkpointer = None
    if _checkpointer_pool is not None:
        try:
            await _checkpointer_pool.close()
        except Exception:
            pass
        _checkpointer_pool = None


async def reset_checkpointer():
    """Reset checkpointer pool and compiled graph when a closed connection is encountered."""
    await close_checkpointer()


async def memory_node(state: ChatState, writer: StreamWriter) -> dict:
    """LangGraph node: tiered conversation memory buffer and non-blocking rolling summary.
    
    Partitions prior history into recent verbatim messages within token budget and
    maintains an incremental rolling summary of older turns without calling
    the LLM in the critical path or blocking user streaming response.
    """
    t_start = time.perf_counter()
    messages = state.get("messages", [])
    existing_summary = state.get("summary")
    summary_through_idx = state.get("summary_through_index") or 0
    model = state.get("model", "openai/gpt-oss-120b")
    thread_id = state.get("thread_id")

    # Extract user/assistant conversational turns (excluding system prompt if already present)
    chat_turns = [m for m in messages if m.get("role") != "system"]

    if len(chat_turns) <= 1:
        overhead_ms = round((time.perf_counter() - t_start) * 1000, 1)
        mem_status = {
            "active": True,
            "summary_status": "Not required",
            "recent_tokens": 0,
            "summary_turns": 0,
            "recall_used": "Not used",
            "prompt_tokens": count_messages_tokens(messages),
        }
        writer({"type": "memory_status", "data": mem_status})
        return {
            "recent_messages": [],
            "summary": existing_summary,
            "summary_through_index": summary_through_idx,
            "memory_overhead_ms": overhead_ms,
            "memory_summary_status": "Not required",
            "memory_recent_tokens": 0,
            "memory_summary_turns": 0,
        }

    # Active user query is the last turn
    prior_turns = chat_turns[:-1]

    # Partition prior turns within configured token budget (e.g. 2048 tokens)
    older_messages, recent_messages = partition_messages(
        prior_messages=prior_turns,
        token_budget=settings.memory_token_budget,
    )
    recent_tokens = count_messages_tokens(recent_messages)

    summary_status = "Not required"
    summary_turns = 0

    # Non-blocking rolling summary: schedule background update if threshold crossed
    if should_update_summary(
        older_messages=older_messages,
        summary_through_index=summary_through_idx,
        threshold_tokens=settings.memory_summary_threshold_tokens,
        min_messages=settings.memory_summary_min_messages,
    ):
        unsummarized = older_messages[summary_through_idx:]
        new_summary_idx = len(older_messages)
        if thread_id:
            logger.info(
                f"[Memory] Offloading rolling summary generation ({len(unsummarized)} turns) "
                f"to background worker for thread {thread_id}..."
            )
            background_summary_manager.schedule_summary(
                thread_id=thread_id,
                existing_summary=existing_summary,
                messages_to_summarize=unsummarized,
                new_summary_idx=new_summary_idx,
                model=model,
            )
        summary_status = "Updated"
        summary_turns = len(older_messages)
    elif existing_summary and existing_summary.strip():
        summary_status = "Available"
        summary_turns = summary_through_idx or len(older_messages)
    else:
        summary_status = "Not required"
        summary_turns = 0

    overhead_ms = round((time.perf_counter() - t_start) * 1000, 1)

    mem_status = {
        "active": True,
        "summary_status": summary_status,
        "recent_tokens": recent_tokens,
        "summary_turns": summary_turns,
        "recall_used": "Not used",
    }
    writer({"type": "memory_status", "data": mem_status})

    return {
        "summary": existing_summary,
        "summary_through_index": summary_through_idx,
        "recent_messages": recent_messages,
        "memory_overhead_ms": overhead_ms,
        "memory_summary_status": summary_status,
        "memory_recent_tokens": recent_tokens,
        "memory_summary_turns": summary_turns,
    }


async def chat_node(state: ChatState, writer: StreamWriter) -> dict:
    """LangGraph node: streams model output and records accumulated response.
    
    Emits custom events via LangGraph's StreamWriter:
    - meta: Provider and model resolution
    - token: Token-by-token deltas
    - memory_status: Runtime conversation memory & token budget details
    - done: Completion status with granular timing telemetry
    - error: Any exception encountered during streaming
    """
    model = state.get("model", "openai/gpt-oss-120b")
    provider = state.get("provider") or resolve_provider(model)
    messages = state.get("messages", [])
    recent_messages = state.get("recent_messages")
    summary = state.get("summary")
    temperature = state.get("temperature", 0.7)

    # Initial meta event via LangGraph StreamWriter
    writer({"type": "meta", "data": {"model": model, "provider": provider}})

    # Deterministic honest refusal when RAG determines query cannot be answered from document
    if (
        state.get("document_ids")
        and not state.get("citations")
        and state.get("retrieved_chunks") == []
        and any("CANNOT be answered" in m.get("content", "") for m in messages if m.get("role") == "system")
    ):
        refusal_text = "The provided document(s) do not contain sufficient information to answer this question."
        writer({"type": "token", "data": {"delta": refusal_text}})
        ret_ms = float(state.get("retrieval_latency_ms") or 0.0)
        rer_ms = float(state.get("rerank_latency_ms") or 0.0)
        writer({
            "type": "done",
            "data": {
                "status": "completed",
                "finish_reason": "stop",
                "timings": {
                    "retrieval_ms": ret_ms,
                    "rerank_ms": rer_ms,
                    "tok_ms": 0.0,
                    "inf_ms": 0.0,
                    "memory_ms": 0.0,
                    "queue_wait_ms": 0.0,
                    "provider_ttft_ms": 0.0,
                    "llm_generation_ms": 0.0,
                    "llm_total_ms": 0.0,
                    "prompt_tokens": 0,
                    "backend_total_ms": round(ret_ms + rer_ms, 1),
                },
                "memory_status": {"active": True},
            }
        })
        return {"response": refusal_text, "error": None, "estimated_prompt_tokens": 0}

    # Assemble contextual payload with protected query and fresh RAG chunks
    chat_turns = [m for m in messages if m.get("role") != "system"]
    active_query = chat_turns[-1] if chat_turns else {"role": "user", "content": ""}

    assembled_messages = []

    # 1. System prompt (Grounding instructions from rerank_node + rolling summary)
    sys_msg = next((m for m in messages if m.get("role") == "system"), None)
    system_text = sys_msg["content"] if sys_msg else (
        "You are TARK Studio's grounded document intelligence assistant.\n"
        "Provide direct, concise, and technically accurate answers with necessary code examples."
    )

    if summary and summary.strip():
        system_text += f"\n\nPREVIOUS CONVERSATION CONTEXT SUMMARY:\n{summary.strip()}"

    assembled_messages.append({"role": "system", "content": system_text})

    # 2. Recent conversation messages buffer (verbatim detail within token budget)
    if recent_messages is not None:
        for m in recent_messages:
            assembled_messages.append({"role": m["role"], "content": m["content"]})
    else:
        # Fallback if memory node was bypassed: include prior turns
        for m in chat_turns[:-1]:
            assembled_messages.append({"role": m["role"], "content": m["content"]})

    # 3. Active user query (always preserved in full verbatim)
    assembled_messages.append(active_query)

    estimated_prompt_tokens = count_messages_tokens(assembled_messages)

    accumulated: List[str] = []
    t_llm_start = time.perf_counter()
    t_first_chunk = None

    try:
        stream = get_model_stream(
            model=model,
            messages=assembled_messages,
            temperature=temperature,
            provider=provider,
            writer=writer,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )
        async for delta in stream:
            if t_first_chunk is None:
                t_first_chunk = time.perf_counter()
            accumulated.append(delta)
            writer({"type": "token", "data": {"delta": delta}})

        t_llm_end = time.perf_counter()
        if t_first_chunk is None:
            t_first_chunk = t_llm_end

        full_text = "".join(accumulated)
        provider_ttft_ms = round((t_first_chunk - t_llm_start) * 1000, 1)
        llm_generation_ms = round((t_llm_end - t_first_chunk) * 1000, 1)
        llm_total_ms = round((t_llm_end - t_llm_start) * 1000, 1)

        retrieval_ms = float(state.get("retrieval_latency_ms") or 0.0)
        rerank_ms = float(state.get("rerank_latency_ms") or 0.0)
        tok_ms = float(state.get("tok_ms") or 0.0)
        inf_ms = float(state.get("inf_ms") or 0.0)
        memory_ms = float(state.get("memory_overhead_ms") or 0.0)
        queue_wait_ms = float(getattr(groq_rate_limiter, "last_queue_wait_ms", 0.0))
        backend_total_ms = round(retrieval_ms + rerank_ms + memory_ms + llm_total_ms, 1)

        memory_status_payload = {
            "active": True,
            "summary_status": state.get("memory_summary_status", "Not required"),
            "recent_tokens": state.get("memory_recent_tokens", 0),
            "summary_turns": state.get("memory_summary_turns", 0),
            "recall_used": "Not used",
            "prompt_tokens": estimated_prompt_tokens,
        }

        timings_payload = {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "tok_ms": tok_ms,
            "inf_ms": inf_ms,
            "memory_ms": memory_ms,
            "queue_wait_ms": queue_wait_ms,
            "provider_ttft_ms": provider_ttft_ms,
            "llm_generation_ms": llm_generation_ms,
            "llm_total_ms": llm_total_ms,
            "prompt_tokens": estimated_prompt_tokens,
            "backend_total_ms": backend_total_ms,
            "rate_limit": getattr(groq_rate_limiter, "last_headers", {}),
        }

        writer({
            "type": "done",
            "data": {
                "status": "completed",
                "finish_reason": "stop",
                "timings": timings_payload,
                "memory_status": memory_status_payload,
            }
        })
        return {"response": full_text, "error": None, "estimated_prompt_tokens": estimated_prompt_tokens}

    except (asyncio.CancelledError, GeneratorExit):
        logger.info(f"LangGraph chat_node execution cancelled for model {model}.")
        raise
    except Exception as exc:
        logger.error(f"Error during LangGraph chat_node execution: {exc}")
        writer({"type": "error", "data": {"error": str(exc)}})
        return {"response": "".join(accumulated), "error": str(exc)}


# ---------------------------------------------------------------------------
# Query intent & off-topic classification
# ---------------------------------------------------------------------------

_BROAD_OVERVIEW_PATTERNS = [
    re.compile(r"\bwhat\s+(is|does|are|was|were)\s+(this|the|a)\s+(pdf|document|book|file|chapter|article)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(is|are|was)\s+in\s+(this|the)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(topics?|subjects?|content|material|information)\b", re.IGNORECASE),
    re.compile(r"\b(summarize|summarise|summary|summarization|overview|introduction|introduction to)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(does|do|did)\s+(it|this|the\s+book|the\s+document|the\s+pdf|the\s+file)\s+(cover|discuss|contain|include|talk about|describe|explain)\b", re.IGNORECASE),
    re.compile(r"\btable\s+of\s+contents\b", re.IGNORECASE),
    re.compile(r"\bbriefly\s+(describe|explain|summarize|summarise|outline)\b", re.IGNORECASE),
    re.compile(r"\bgive\s+(me\s+)?(an?\s+)?(overview|summary|introduction|outline|description)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+can\s+(i|we)\s+(learn|find|read)\b", re.IGNORECASE),
    re.compile(r"\bmain\s+(topics?|themes?|ideas?|concepts?|chapters?|sections?)\b", re.IGNORECASE),
]


def _is_broad_document_query(query: str) -> bool:
    """Return True when the query is asking for a document overview or summary."""
    q = query.strip()
    if len(q.split()) <= 3 and any(kw in q.lower() for kw in ("pdf", "document", "book", "file", "about")):
        return True
    return any(pat.search(q) for pat in _BROAD_OVERVIEW_PATTERNS)


def evaluate_is_off_topic(
    query: str,
    chunks: List[Any],
    candidate_pool_size: int = 0,
    reranker_type: str = "onnx",
    ce_onnx_threshold: float = -9.5,
) -> bool:
    """Evaluate whether a query is truly off-topic / unsupported for the retrieved chunks.
    
    Multi-Channel Evidence Evaluation:
    Evaluates evidence across three retrieval channels:
    1. Cross-Encoder Logit (Relevance): ONNX cross-encoder produces uncalibrated logits
       ranging from ~ -11.5 (floor) to +10.0 (exact answer). On empirical benchmarks:
       - In-domain queries on large corpora score >= -3.9 (mean +0.9).
       - In-domain queries on compact semi-structured docs score >= -6.3 (except tabular headers ~ -10.3).
       - Unrelated off-topic queries score <= -10.4 (mean -11.0).
       A score >= -9.5 provides strong evidence of query-passage relevance.
    2. Dense Semantic Similarity: Cosine similarity in embedding space.
       - For compact documents (<= 10 chunks), in-domain queries score >= 0.48.
       - For large corpora (> 10 chunks), empirical random tail similarity can reach ~0.47,
         so dense similarity alone requires >= 0.55 or corroborating lexical/reranker signal.
    3. Sparse Lexical Match: Full-text search ts_rank_cd.
       - A positive sparse match alone is NOT sufficient if semantic relevance is near zero
         (preventing incidental keyword collisions from falsely accepting off-topic queries).
       - Sparse match requires corroboration (dense >= 0.40 or CE >= -10.0).
    
    A query is classified as unsupported (off-topic) if:
    1. Candidate pool is empty; OR
    2. Cross-encoder relevance is near the model floor (CE < threshold) AND dense similarity
       fails to exceed the operating threshold AND sparse matches lack semantic corroboration.
    """
    if not chunks:
        return True

    pool_size = candidate_pool_size or len(chunks)
    norm_type = (reranker_type or "").lower()
    ce_threshold = -10.0 if pool_size <= 10 else ce_onnx_threshold

    max_ce = max((getattr(c, "rerank_score", None) or -99.0 for c in chunks), default=-99.0)
    max_dense = max((getattr(c, "dense_score", None) or 0.0 for c in chunks), default=0.0)
    has_sparse = any((getattr(c, "sparse_score", None) or 0.0) > 0 for c in chunks)

    # Condition 1: Strong Cross-Encoder signal (semantic QA match)
    if max_ce >= ce_threshold:
        return False

    # Condition 2: Corroborated Lexical match
    # Sparse match must be supported by non-noise dense similarity (>= 0.35) or non-floor CE (>= -10.5)
    if has_sparse and (max_dense >= 0.35 or max_ce >= -10.5):
        return False

    # Condition 3: Dense semantic alignment
    dense_threshold = 0.42 if pool_size <= 10 else 0.52
    if max_dense >= dense_threshold:
        return False

    # All evidence channels indicate the query is unsupported / off-topic
    return True


async def retrieve_node(state: ChatState, writer: StreamWriter) -> dict:
    """LangGraph node: executes conversation-aware hybrid retrieval (vector + FTS with RRF) for candidates."""
    document_ids = state.get("document_ids")
    messages = state.get("messages", [])

    if not document_ids or not messages:
        return {}

    user_query = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    if not user_query:
        return {}

    try:
        from app.rag.retriever import hybrid_retriever
        from app.config import Settings
        from app.rag.query_resolver import resolve_conversational_query

        settings = Settings()

        doc_uuids = [uuid.UUID(d) if isinstance(d, str) else d for d in document_ids]
        candidate_k = settings.reranker_candidate_k or 20

        # Retrieve attached document filenames for context resolution
        doc_names = []
        try:
            from app.database import async_session_factory, Document
            from sqlalchemy import select
            async with async_session_factory() as session:
                res = await session.execute(select(Document.filename).where(Document.id.in_(doc_uuids)))
                doc_names = [row[0] for row in res.all()]
        except Exception:
            pass

        # Resolve conversational pronouns, references, and intents
        resolved = resolve_conversational_query(query=user_query, messages=messages, doc_names=doc_names)

        t_ret_start = time.perf_counter()
        # Retrieve candidates via hybrid retrieval with resolved query
        chunks = await hybrid_retriever.retrieve(
            query=user_query,
            document_ids=doc_uuids,
            top_k=candidate_k,
            max_context_tokens=4000,
            resolved_query=resolved.search_query,
        )
        retrieval_latency_ms = round((time.perf_counter() - t_ret_start) * 1000, 1)

        return {
            "candidate_chunks": chunks,
            "retrieval_latency_ms": retrieval_latency_ms,
            "resolved_query": resolved.search_query,
            "is_followup": resolved.is_followup,
            "is_correction": resolved.is_correction,
            "dense_candidates_count": hybrid_retriever.last_dense_count,
            "sparse_candidates_count": hybrid_retriever.last_sparse_count,
            "fused_candidates_count": hybrid_retriever.last_fused_count,
        }

    except Exception as exc:
        logger.error(f"Error in retrieve_node: {exc}")
        writer({"type": "sources", "data": {"citations": [], "error": str(exc)}})
        return {"candidate_chunks": [], "error": str(exc), "retrieval_latency_ms": 0.0}


async def rerank_node(state: ChatState, writer: StreamWriter) -> dict:
    """LangGraph node: reranks retrieved candidates using configured cross-encoder reranker."""
    candidate_chunks = state.get("candidate_chunks", [])
    messages = state.get("messages", [])

    if not messages:
        return {}

    user_query = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    if not user_query:
        return {}

    from app.config import Settings
    from app.rag.reranker import get_reranker

    settings = Settings()
    reranker_type = state.get("reranker_type") or settings.reranker_type or "baseline"
    retrieval_latency_ms = float(state.get("retrieval_latency_ms") or 0.0)

    try:
        reranker = get_reranker(reranker_type)

        # Classify broad overview query
        broad_query = _is_broad_document_query(user_query)

        # Adaptive top_k evidence selection:
        # 1. Compact candidate sets (<= 10 chunks total, e.g. single-page or small documents):
        #    Preserve all candidates so no vital facts, attributes, or sections are dropped.
        # 2. Comprehensive extraction queries (lists, all items, breakdowns, overviews):
        #    Preserve up to 10 chunks to ensure full coverage within context token budget.
        # 3. Standard focused queries on large corpora:
        #    Retain top 5 chunks for focused, low-noise context.
        is_comprehensive_query = bool(re.search(
            r"\b(all|every|list|mention|complete|full|entire|details?|breakdown|outline|compare|differences?)\b",
            user_query,
            re.IGNORECASE,
        ))
        if broad_query or is_comprehensive_query or len(candidate_chunks) <= 10:
            adaptive_top_k = min(len(candidate_chunks), 10)
        else:
            adaptive_top_k = min(len(candidate_chunks), settings.reranker_top_k or 5)

        ranking_query = state.get("resolved_query") or user_query

        t_rerank_start = time.perf_counter()
        # Non-blocking: offload cross-encoder scoring to thread pool to preserve event loop responsiveness
        reranked_chunks = await asyncio.to_thread(
            reranker.rerank,
            query=ranking_query,
            chunks=candidate_chunks,
            top_k=adaptive_top_k,
        )
        rerank_latency_ms = round((time.perf_counter() - t_rerank_start) * 1000, 1)

        # Sensible fallback: preserve best hybrid candidates if reranker returns empty
        if not reranked_chunks and candidate_chunks:
            logger.warning("Reranker returned empty chunks; applying sensible fallback to base hybrid candidates.")
            reranked_chunks = candidate_chunks[:adaptive_top_k]

        tok_ms = getattr(reranker, "last_tok_ms", 0.0)
        inf_ms = getattr(reranker, "last_inf_ms", 0.0)

        norm_type = reranker_type.lower()
        if norm_type in ("baseline", "none"):
            reranker_display = "OFF"
        elif "onnx" in norm_type:
            reranker_display = "ONNX"
        else:
            reranker_display = reranker_type.upper()

        logger.info(
            f"[RAG] LangGraph rerank_node: reranker='{reranker_display}', "
            f"candidates_in={len(candidate_chunks)}, reranked_out={len(reranked_chunks)}, "
            f"latency={rerank_latency_ms}ms (tok={tok_ms}ms, inf={inf_ms}ms)"
        )

        # Count structured links and references across candidate evidence
        structured_links_count = sum(
            c.content.count("http://") + c.content.count("https://") + c.content.count("mailto:")
            for c in reranked_chunks
        )

        # Check for genuine out-of-document honest refusal
        # Cross-encoder reorders candidates; evaluate_is_off_topic evaluates multi-channel consensus
        is_off_topic = evaluate_is_off_topic(
            query=ranking_query,
            chunks=reranked_chunks,
            candidate_pool_size=len(candidate_chunks),
            reranker_type=reranker_type,
        )

        rag_status = {
            "retrieval": "Hybrid",
            "reranker": reranker_display,
            "candidates_count": len(candidate_chunks),
            "reranked_count": len(reranked_chunks) if not is_off_topic else 0,
            "dense_candidates_count": int(state.get("dense_candidates_count") or 0),
            "sparse_candidates_count": int(state.get("sparse_candidates_count") or 0),
            "fused_candidates_count": int(state.get("fused_candidates_count") or len(candidate_chunks)),
            "candidates_before_rerank": len(candidate_chunks),
            "candidates_after_rerank": len(reranked_chunks),
            "evidence_selected_count": len(reranked_chunks) if not is_off_topic else 0,
            "evidence_rejected_count": max(0, len(candidate_chunks) - (len(reranked_chunks) if not is_off_topic else 0)),
            "evidence_rejected_reason": "off_topic_threshold" if is_off_topic else "top_k_selection",
            "structured_links_count": structured_links_count,
            "retrieval_latency_ms": retrieval_latency_ms,
            "rerank_latency_ms": rerank_latency_ms,
            "tok_ms": tok_ms,
            "inf_ms": inf_ms,
            "query_type": "broad_overview" if broad_query else "specific",
            "is_followup": bool(state.get("is_followup")),
            "is_correction": bool(state.get("is_correction")),
        }

        if is_off_topic:
            # Emit zero citations - do not provide irrelevant source cards
            rag_status["reranked_count"] = 0
            writer({"type": "sources", "data": {"citations": [], "rag_status": rag_status}})

            refusal_instruction = (
                "You are TARK Studio's grounded document intelligence assistant.\n"
                "The user's question CANNOT be answered from the provided knowledge documents.\n"
                "You must state clearly and directly:\n"
                "'The provided document(s) do not contain sufficient information to answer this question.'\n"
                "Do NOT attempt to guess, do NOT recite external facts, and do NOT emit any citations."
            )
            augmented_messages = [
                {"role": "system", "content": refusal_instruction},
                *messages,
            ]
            return {
                "messages": augmented_messages,
                "citations": [],
                "retrieved_chunks": [],
            }

        # Format verified citations and context excerpts
        citations = []
        context_parts = []

        for c in reranked_chunks:
            citation_item = {
                "document_id": str(c.document_id),
                "document_name": c.document_name,
                "viewer_page": c.viewer_page,
                "book_page": c.book_page,
                "citation_label": c.citation_label,
                "heading": c.heading,
                "excerpt": c.content[:1500],
            }
            citations.append(citation_item)
            context_parts.append(f"[{c.citation_label}]\n{c.content}")

        # Emit sources event to client via StreamWriter
        writer({"type": "sources", "data": {"citations": citations, "rag_status": rag_status}})

        if broad_query:
            # For broad document overview queries, guide the LLM to synthesise a
            # high-level answer from the retrieved passages instead of hunting for a
            # single specific fact (which is what the standard grounding prompt does).
            doc_name = reranked_chunks[0].document_name if reranked_chunks else "the document"
            grounding_instruction = (
                f"You are TARK Studio's grounded document intelligence assistant.\n"
                f"The user is asking for an overview or summary of '{doc_name}'.\n"
                "Using ONLY the provided document excerpts below, give a comprehensive, "
                "well-structured answer that covers the document's main topics, key concepts, "
                "and overall structure. If a table of contents, introduction, or chapter headings "
                "are present in the excerpts, use them to organise your response.\n"
                "Cite every factual statement with its source label, e.g. [Page X (PDF p. Y)].\n"
                "Do NOT fabricate information not found in the excerpts.\n\n"
                "DOCUMENT CONTEXT EXCERPTS:\n"
                + "\n\n---\n\n".join(context_parts)
            )
        else:
            grounding_instruction = (
                "You are TARK Studio's grounded document intelligence assistant.\n"
                "Answer the user's question accurately, thoroughly, and strictly using ONLY the provided document excerpts below.\n"
                "Every factual statement must cite its source in the format [Page X (PDF p. Y)] or [Page X].\n\n"
                "CRITICAL SOURCE-OF-TRUTH HIERARCHY & GROUNDING RULES:\n"
                "1. DOCUMENT EVIDENCE IS THE SOLE AUTHORITATIVE SOURCE OF TRUTH:\n"
                "   Base all factual answers strictly on the extracted text, hyperlinks, contact details, and tables present in the excerpts below.\n"
                "2. CONVERSATION CONTEXT VS DOCUMENT FACTS:\n"
                "   Conversation history is provided ONLY to resolve pronouns (he, she, it) and user intent.\n"
                "   NEVER adopt or repeat factual claims, locations, dates, or numbers from prior conversation turns if they are not explicitly substantiated by the current document excerpts.\n"
                "3. CONFLICT RESOLUTION & RE-VERIFICATION:\n"
                "   - If an earlier assistant answer or user message mentioned an incorrect or contradictory fact, supersede it completely by stating the true fact strictly from the current document evidence with its exact citation.\n"
                "   - If the document contains multiple distinct values (such as primary contact vs secondary references), distinguish each context clearly with its respective citation rather than guessing.\n"
                "4. FACTUAL ATTRIBUTE EXTRACTION:\n"
                "   Extract all specific names, locations, LinkedIn/GitHub/portfolio URLs, contact emails/phones, dates, numbers, skills, and table records present in the text.\n"
                "5. If the provided document excerpts do not contain the answer, explicitly state:\n"
                "'The provided document(s) do not contain sufficient information to answer this question.'\n"
                "NEVER fabricate citations or make claims unsupported by the text.\n\n"
                "DOCUMENT CONTEXT EXCERPTS:\n"
                + "\n\n---\n\n".join(context_parts)
            )

        augmented_messages = [
            {"role": "system", "content": grounding_instruction},
            *messages,
        ]

        return {
            "messages": augmented_messages,
            "citations": citations,
            "retrieved_chunks": [c.__dict__ for c in reranked_chunks],
            "retrieval_latency_ms": retrieval_latency_ms,
            "rerank_latency_ms": rerank_latency_ms,
            "tok_ms": tok_ms,
            "inf_ms": inf_ms,
        }

    except Exception as exc:
        logger.error(f"Error in rerank_node with reranker '{reranker_type}': {exc}")
        writer({"type": "error", "data": {"error": f"Reranker '{reranker_type}' execution failed: {exc}"}})
        raise RuntimeError(f"Reranker '{reranker_type}' failed: {exc}. Refusing silent fallback.") from exc


def create_state_graph() -> StateGraph:
    """Instantiate uncompiled StateGraph definition with dynamic RAG and memory routing."""
    builder = StateGraph(ChatState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("memory", memory_node)
    builder.add_node("chat", chat_node)

    def route_start(state: ChatState) -> str:
        if state.get("document_ids"):
            return "retrieve"
        return "memory"

    builder.add_conditional_edges(START, route_start, {"retrieve": "retrieve", "memory": "memory"})
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "memory")
    builder.add_edge("memory", "chat")
    builder.add_edge("chat", END)
    return builder


async def get_compiled_graph():
    """Return pre-compiled StateGraph with active checkpointer, avoiding per-request compile overhead."""
    global _active_compiled_graph
    if _active_compiled_graph is not None:
        return _active_compiled_graph
    checkpointer = await get_checkpointer()
    builder = create_state_graph()
    _active_compiled_graph = builder.compile(checkpointer=checkpointer)
    return _active_compiled_graph


def build_chat_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Compile the LangGraph chat execution graph with explicit checkpointer."""
    builder = create_state_graph()
    return builder.compile(checkpointer=checkpointer or MemorySaver())


# Reference chat graph compiled with explicit MemorySaver for isolated unit testing
chat_graph = build_chat_graph(checkpointer=MemorySaver())


async def stream_chat_workflow(
    messages: List[dict],
    model: str,
    temperature: float = 0.7,
    thread_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    request: Optional[Request] = None,
    document_ids: Optional[List[str]] = None,
    reranker_type: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Execute chat workflow through LangGraph with AsyncPostgresSaver checkpointing.
    
    Uses conversation UUID as the LangGraph thread_id.
    """
    provider = resolve_provider(model)
    effective_thread_id = thread_id or str(uuid.uuid4())
    effective_conv_id = conversation_id or effective_thread_id

    initial_state: ChatState = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "provider": provider,
        "response": "",
        "error": None,
        "document_ids": document_ids,
        "reranker_type": reranker_type,
        "thread_id": effective_thread_id,
    }

    # Use pre-compiled graph to eliminate per-request compilation latency
    compiled_graph = await get_compiled_graph()

    config = {
        "configurable": {
            "thread_id": effective_thread_id,
        }
    }

    retried = False
    while True:
        yielded_any = False
        try:
            async for chunk in compiled_graph.astream(
                initial_state,
                stream_mode="custom",
                config=config,
            ):
                # Check for client abort / disconnect
                if request and await request.is_disconnected():
                    logger.info(f"Client disconnected; halting LangGraph thread {effective_thread_id}.")
                    return

                event_type = chunk.get("type", "token")
                data_dict = chunk.get("data", {})

                # Enrich meta event with conversation_id for client tracking
                if event_type == "meta":
                    data_dict["conversation_id"] = effective_conv_id

                data_str = json.dumps(data_dict)
                yield f"event: {event_type}\ndata: {data_str}\n\n"
                yielded_any = True
            return

        except Exception as exc:
            err_msg = str(exc).lower()
            is_conn_error = (
                "connection is closed" in err_msg
                or "closed connection" in err_msg
                or "connection already closed" in err_msg
                or "operationalerror" in err_msg
                or "broken pipe" in err_msg
            )
            if not retried and not yielded_any and is_conn_error:
                logger.warning(
                    f"PostgreSQL connection drop detected ({exc}); "
                    "resetting checkpointer pool and transparently retrying stream..."
                )
                await reset_checkpointer()
                compiled_graph = await get_compiled_graph()
                retried = True
                continue

            logger.error(f"Unexpected error in LangGraph stream execution: {exc}")
            err_str = json.dumps({"error": str(exc), "conversation_id": effective_conv_id})
            yield f"event: error\ndata: {err_str}\n\n"
            return


# ==============================================================================
# LangGraph Workflow: Autonomous Chat Title Generation
# ==============================================================================

async def title_node(state: TitleState) -> dict:
    """LangGraph node: generate a concise, descriptive title (3-5 words) for a conversation."""
    prompt = state.get("prompt", "").strip()
    model = state.get("model", "openai/gpt-oss-120b")
    provider = state.get("provider") or resolve_provider(model)

    if not prompt:
        return {"title": "New Chat", "error": None}

    sys_instruction = (
        "You are an expert conversational title generator. "
        "Generate a concise, descriptive, editorial title (3 to 5 words max) summarizing the conversation starter. "
        "Do NOT use quotes, punctuation, or preamble. Return ONLY the title text."
    )
    user_query = f"Conversation starter:\n{prompt[:300]}"

    try:
        stream = get_model_stream(
            model=model,
            messages=[
                {"role": "system", "content": sys_instruction},
                {"role": "user", "content": user_query},
            ],
            temperature=0.3,
            provider=provider,
        )
        tokens = []
        async for chunk in stream:
            tokens.append(chunk)
        raw_title = "".join(tokens).strip().strip('"').strip("'").strip()
        clean_title = raw_title.split("\n")[0].strip()
        if not clean_title or len(clean_title) < 2:
            clean_title = (prompt[:30] + "...") if len(prompt) > 30 else prompt
        return {"title": clean_title[:60], "error": None}
    except Exception as exc:
        logger.error(f"Error in LangGraph title_node: {exc}")
        fallback = (prompt[:30] + "...") if len(prompt) > 30 else prompt
        return {"title": fallback, "error": str(exc)}


def build_title_graph():
    """Compile LangGraph workflow for autonomous chat title generation."""
    builder = StateGraph(TitleState)
    builder.add_node("generate_title", title_node)
    builder.add_edge(START, "generate_title")
    builder.add_edge("generate_title", END)
    return builder.compile()


title_graph = build_title_graph()


async def generate_chat_title(prompt: str, model: str = "openai/gpt-oss-120b") -> str:
    """Invoke the LangGraph title generation workflow."""
    provider = resolve_provider(model)
    initial_state: TitleState = {
        "prompt": prompt,
        "model": model,
        "provider": provider,
        "title": "",
        "error": None,
    }
    result = await title_graph.ainvoke(initial_state)
    return result.get("title") or "New Chat"
