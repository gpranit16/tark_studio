"""LangGraph AI Orchestration for TARK Studio chat workflows with AsyncPostgresSaver."""

import sys
import json
import logging
import uuid
import asyncio
import time
from typing import AsyncGenerator, List, Optional
from fastapi import Request
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings
from app.ai.state import ChatState, TitleState
from app.ai.providers import get_model_stream, resolve_provider

logger = logging.getLogger(__name__)

# Checkpointer singletons
_checkpointer_cm = None
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
    return (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}@"
        f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


async def get_checkpointer() -> BaseCheckpointSaver:
    """Return persistent AsyncPostgresSaver by default.
    
    Raises RuntimeError if PostgreSQL is unavailable.
    Silent fallback to MemorySaver is forbidden; MemorySaver is only permitted
    when explicitly configured via set_test_checkpointer().
    """
    global _checkpointer_cm, _postgres_checkpointer, _test_checkpointer

    # Explicit test checkpointer permitted only if explicitly registered
    if _test_checkpointer is not None:
        return _test_checkpointer

    if _postgres_checkpointer is not None:
        return _postgres_checkpointer

    try:
        conn_string = get_postgres_connection_string()
        _checkpointer_cm = AsyncPostgresSaver.from_conn_string(conn_string)
        _postgres_checkpointer = await _checkpointer_cm.__aenter__()
        await _postgres_checkpointer.setup()
        logger.info("LangGraph AsyncPostgresSaver initialized successfully.")
        return _postgres_checkpointer
    except Exception as exc:
        logger.error(f"PostgreSQL checkpointer initialization failed: {exc}")
        raise RuntimeError(
            f"PostgreSQL checkpointer is unavailable: {exc}. "
            "PostgreSQL checkpointing is required for TARK Studio session state. "
            "Silent fallback to in-memory storage is disabled."
        )


async def close_checkpointer():
    """Dispose of checkpointer connection pool during shutdown."""
    global _checkpointer_cm, _postgres_checkpointer, _active_compiled_graph, _test_checkpointer
    _active_compiled_graph = None
    _test_checkpointer = None
    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.__aexit__(None, None, None)
        except Exception:
            pass
        _checkpointer_cm = None
        _postgres_checkpointer = None


async def chat_node(state: ChatState, writer: StreamWriter) -> dict:
    """LangGraph node: streams model output and records accumulated response.
    
    Emits custom events via LangGraph's StreamWriter:
    - meta: Provider and model resolution
    - token: Token-by-token deltas
    - done: Completion status
    - error: Any exception encountered during streaming
    """
    model = state.get("model", "openai/gpt-oss-120b")
    provider = state.get("provider") or resolve_provider(model)
    messages = state.get("messages", [])
    temperature = state.get("temperature", 0.7)

    # Initial meta event via LangGraph StreamWriter
    writer({"type": "meta", "data": {"model": model, "provider": provider}})

    accumulated: List[str] = []
    try:
        stream = get_model_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            provider=provider,
        )
        async for delta in stream:
            accumulated.append(delta)
            writer({"type": "token", "data": {"delta": delta}})

        full_text = "".join(accumulated)
        writer({"type": "done", "data": {"status": "completed", "finish_reason": "stop"}})
        return {"response": full_text, "error": None}

    except (asyncio.CancelledError, GeneratorExit):
        logger.info(f"LangGraph chat_node execution cancelled for model {model}.")
        raise
    except Exception as exc:
        logger.error(f"Error during LangGraph chat_node execution: {exc}")
        writer({"type": "error", "data": {"error": str(exc)}})
        return {"response": "".join(accumulated), "error": str(exc)}


async def retrieve_node(state: ChatState, writer: StreamWriter) -> dict:
    """LangGraph node: executes base hybrid retrieval (vector + FTS with RRF) for candidates."""
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
        settings = Settings()

        doc_uuids = [uuid.UUID(d) if isinstance(d, str) else d for d in document_ids]
        candidate_k = settings.reranker_candidate_k or 20

        # Retrieve candidates via unchanged base dense + FTS + RRF
        chunks = await hybrid_retriever.retrieve(
            query=user_query,
            document_ids=doc_uuids,
            top_k=candidate_k,
            max_context_tokens=4000,
        )

        return {"candidate_chunks": chunks}

    except Exception as exc:
        logger.error(f"Error in retrieve_node: {exc}")
        writer({"type": "sources", "data": {"citations": [], "error": str(exc)}})
        return {"candidate_chunks": [], "error": str(exc)}


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

    try:
        t_rerank_start = time.perf_counter()
        reranker = get_reranker(reranker_type)
        top_k = settings.reranker_top_k or 5

        # Rerank candidates using cross-encoder
        reranked_chunks = reranker.rerank(
            query=user_query,
            chunks=candidate_chunks,
            top_k=top_k,
        )
        rerank_latency_ms = round((time.perf_counter() - t_rerank_start) * 1000, 1)

        norm_type = reranker_type.lower()
        if norm_type in ("baseline", "none"):
            reranker_display = "OFF"
        elif "onnx" in norm_type:
            reranker_display = "ONNX"
        elif "bge" in norm_type:
            reranker_display = "BGE"
        else:
            reranker_display = reranker_type.upper()

        logger.info(
            f"[RAG] LangGraph rerank_node: reranker='{reranker_display}', "
            f"candidates_in={len(candidate_chunks)}, reranked_out={len(reranked_chunks)}, "
            f"latency={rerank_latency_ms}ms"
        )

        rag_status = {
            "retrieval": "Hybrid",
            "reranker": reranker_display,
            "candidates_count": len(candidate_chunks),
            "reranked_count": len(reranked_chunks),
            "rerank_latency_ms": rerank_latency_ms,
        }

        # Check for out-of-document honest refusal
        # If cross-encoder scores are strongly negative or no chunks returned, refuse without citations
        is_off_topic = False
        if not reranked_chunks:
            is_off_topic = True
        elif reranker_type.lower() == "bge":
            max_score = max((getattr(c, "rerank_score", None) or -99.0 for c in reranked_chunks), default=-99.0)
            if max_score < -5.0:
                is_off_topic = True
        elif reranker_type.lower() == "onnx":
            max_score = max((getattr(c, "rerank_score", None) or -99.0 for c in reranked_chunks), default=-99.0)
            if max_score < -8.5:
                is_off_topic = True

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

        grounding_instruction = (
            "You are TARK Studio's grounded document intelligence assistant.\n"
            "Answer the user's question using ONLY the provided document excerpts below.\n"
            "Every factual statement must cite its source in the format [Page X (PDF p. Y)] or [Page X].\n"
            "Strict Domain Rules:\n"
            "1. Regular Expressions: In Python's re module, '.*?' is non-greedy/lazy, but character class repetitions like '[0-9]+' or '+' remain greedy unless explicitly modified with '?'.\n"
            "2. Relational Databases (SQLite): In Python's sqlite3 and Chapter 15, foreign keys are logical keys and are NOT enforced by SQLite by default unless explicitly activated via PRAGMA foreign_keys = ON;. Do not claim database-enforced referential integrity unless stated in the text.\n"
            "3. Figures and Diagrams: Base answers strictly on the extracted textual captions and labels present in the excerpts. Do NOT claim visual perception or graphic recognition.\n"
            "If the provided document excerpts do not contain the answer, explicitly state:\n"
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
        }

    except Exception as exc:
        logger.error(f"Error in rerank_node with reranker '{reranker_type}': {exc}")
        writer({"type": "error", "data": {"error": f"Reranker '{reranker_type}' execution failed: {exc}"}})
        raise RuntimeError(f"Reranker '{reranker_type}' failed: {exc}. Refusing silent fallback.") from exc


def create_state_graph() -> StateGraph:
    """Instantiate uncompiled StateGraph definition with dynamic RAG routing."""
    builder = StateGraph(ChatState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("chat", chat_node)

    def route_start(state: ChatState) -> str:
        if state.get("document_ids"):
            return "retrieve"
        return "chat"

    builder.add_conditional_edges(START, route_start, {"retrieve": "retrieve", "chat": "chat"})
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "chat")
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
    }

    # Use pre-compiled graph to eliminate per-request compilation latency
    compiled_graph = await get_compiled_graph()

    config = {
        "configurable": {
            "thread_id": effective_thread_id,
        }
    }

    try:
        async for chunk in compiled_graph.astream(
            initial_state,
            stream_mode="custom",
            config=config,
        ):
            # Check for client abort / disconnect
            if request and await request.is_disconnected():
                logger.info(f"Client disconnected; halting LangGraph thread {effective_thread_id}.")
                break

            event_type = chunk.get("type", "token")
            data_dict = chunk.get("data", {})

            # Enrich meta event with conversation_id for client tracking
            if event_type == "meta":
                data_dict["conversation_id"] = effective_conv_id

            data_str = json.dumps(data_dict)
            yield f"event: {event_type}\ndata: {data_str}\n\n"

    except Exception as exc:
        logger.error(f"Unexpected error in LangGraph stream execution: {exc}")
        err_str = json.dumps({"error": str(exc), "conversation_id": effective_conv_id})
        yield f"event: error\ndata: {err_str}\n\n"


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
