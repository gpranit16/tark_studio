"""FastAPI API routes for AI chat, streaming, and persistent conversation management."""

import json
import time
import uuid
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select, desc, func, or_
from sqlalchemy.orm import selectinload

from app.database import async_session_factory, Conversation, Message
from app.schemas.chat import (
    ChatStreamRequest,
    ConversationCreateRequest,
    ConversationUpdateRequest,
    ConversationSummaryResponse,
    ConversationDetailResponse,
    MessageResponse,
    MessageUpdateRequest,
    TruncateMessagesRequest,
    GenerateTitleResponse,
    ModelsResponse,
)
from app.config import settings
from app.ai.providers import SUPPORTED_MODELS, resolve_provider
from app.ai.graph import stream_chat_workflow, generate_chat_title
from app.rag.cache import redis_cache, normalize_query
from app.rag.embeddings import embeddings_client

router = APIRouter(prefix="/api/v1", tags=["Chat & Conversations"])
logger = logging.getLogger(__name__)
_active_cache_tasks: set[asyncio.Task] = set()


def _schedule_cache_write(coro) -> None:
    """Schedule write-behind Redis task while holding a strong reference until completion."""
    task = asyncio.create_task(coro)
    _active_cache_tasks.add(task)
    task.add_done_callback(_active_cache_tasks.discard)


# ==============================================================================
# Model Exploration
# ==============================================================================

@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available AI models across providers."""
    return ModelsResponse(
        models=SUPPORTED_MODELS,
        default="openai/gpt-oss-120b",
    )


# ==============================================================================
# Conversation Management (CRUD)
# ==============================================================================

@router.get("/conversations", response_model=List[ConversationSummaryResponse])
async def list_conversations(
    q: Optional[str] = Query(None, description="Search term for conversation title or message content"),
    is_archived: bool = Query(False, description="Whether to list archived conversations"),
):
    """List saved conversations with PostgreSQL-powered title & message search and archive filtering."""
    async with async_session_factory() as session:
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.is_archived == is_archived)
            .order_by(desc(Conversation.updated_at))
        )

        if q and q.strip():
            clean_q = f"%{q.strip()}%"
            msg_subquery = (
                select(Message.conversation_id)
                .where(Message.content.ilike(clean_q))
            )
            stmt = stmt.where(
                or_(
                    Conversation.title.ilike(clean_q),
                    Conversation.id.in_(msg_subquery),
                )
            )

        result = await session.execute(stmt)
        conversations = result.scalars().all()

        summaries = []
        for conv in conversations:
            msg_count = len(conv.messages)
            last_msg = conv.messages[-1].content if msg_count > 0 else None
            # Truncate preview
            preview = (last_msg[:80] + "...") if last_msg and len(last_msg) > 80 else last_msg
            summaries.append(
                ConversationSummaryResponse(
                    id=conv.id,
                    title=conv.title,
                    model_id=conv.model_id,
                    provider=conv.provider,
                    is_archived=conv.is_archived,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    message_count=msg_count,
                    last_message_preview=preview,
                )
            )
        return summaries


@router.post("/conversations", response_model=ConversationDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(payload: ConversationCreateRequest):
    """Create a new persistent conversation."""
    async with async_session_factory() as session:
        provider = payload.provider or resolve_provider(payload.model_id or "openai/gpt-oss-120b")
        conv = Conversation(
            id=uuid.uuid4(),
            title=payload.title or "New Chat",
            model_id=payload.model_id or "openai/gpt-oss-120b",
            provider=provider,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(conv)
        await session.commit()
        await session.refresh(conv)

        return ConversationDetailResponse(
            id=conv.id,
            title=conv.title,
            model_id=conv.model_id,
            provider=conv.provider,
            is_archived=conv.is_archived,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            messages=[],
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(conversation_id: uuid.UUID):
    """Load conversation and all chronologically ordered messages."""
    async with async_session_factory() as session:
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        result = await session.execute(stmt)
        conv = result.scalar_one_or_none()

        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            )

        messages_sorted = sorted(conv.messages, key=lambda m: m.created_at)

        # Retrieve summary state from LangGraph checkpoint if available
        checkpoint_summary: Optional[str] = None
        checkpoint_summary_idx: Optional[int] = None
        try:
            from app.ai.graph import get_compiled_graph
            graph = await get_compiled_graph()
            snap = await graph.aget_state({"configurable": {"thread_id": str(conversation_id)}})
            if snap and snap.values:
                checkpoint_summary = snap.values.get("summary")
                checkpoint_summary_idx = snap.values.get("summary_through_index")
        except Exception:
            pass

        return ConversationDetailResponse(
            id=conv.id,
            title=conv.title,
            model_id=conv.model_id,
            provider=conv.provider,
            is_archived=conv.is_archived,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            summary=checkpoint_summary,
            summary_through_index=checkpoint_summary_idx,
            messages=[
                MessageResponse(
                    id=m.id,
                    conversation_id=m.conversation_id,
                    role=m.role,
                    content=m.content,
                    model=m.model,
                    provider=m.provider,
                    latency_ms=m.latency_ms,
                    created_at=m.created_at,
                )
                for m in messages_sorted
            ],
        )


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummaryResponse)
async def rename_conversation(conversation_id: uuid.UUID, payload: ConversationUpdateRequest):
    """Rename a conversation."""
    async with async_session_factory() as session:
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await session.execute(stmt)
        conv = result.scalar_one_or_none()

        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            )

        conv.title = payload.title.strip()
        conv.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(conv)

        return ConversationSummaryResponse(
            id=conv.id,
            title=conv.title,
            model_id=conv.model_id,
            provider=conv.provider,
            is_archived=conv.is_archived,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=0,
            last_message_preview=None,
        )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_200_OK)
async def delete_conversation(conversation_id: uuid.UUID):
    """Delete a conversation and its cascading messages."""
    async with async_session_factory() as session:
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await session.execute(stmt)
        conv = result.scalar_one_or_none()

        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            )

        await session.delete(conv)
        await session.commit()

        return {"deleted": True, "id": str(conversation_id)}


@router.post("/conversations/{conversation_id}/archive", response_model=ConversationSummaryResponse)
async def archive_conversation(conversation_id: uuid.UUID):
    """Archive a conversation so it moves out of active chat view."""
    async with async_session_factory() as session:
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await session.execute(stmt)
        conv = result.scalar_one_or_none()

        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            )

        conv.is_archived = True
        conv.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(conv)

        return ConversationSummaryResponse(
            id=conv.id,
            title=conv.title,
            model_id=conv.model_id,
            provider=conv.provider,
            is_archived=conv.is_archived,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=0,
            last_message_preview=None,
        )


@router.post("/conversations/{conversation_id}/unarchive", response_model=ConversationSummaryResponse)
async def unarchive_conversation(conversation_id: uuid.UUID):
    """Restore an archived conversation back to active chat view."""
    async with async_session_factory() as session:
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await session.execute(stmt)
        conv = result.scalar_one_or_none()

        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            )

        conv.is_archived = False
        conv.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(conv)

        return ConversationSummaryResponse(
            id=conv.id,
            title=conv.title,
            model_id=conv.model_id,
            provider=conv.provider,
            is_archived=conv.is_archived,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=0,
            last_message_preview=None,
        )


@router.post("/conversations/{conversation_id}/generate-title", response_model=GenerateTitleResponse)
async def generate_title_endpoint(conversation_id: uuid.UUID):
    """Trigger LangGraph workflow to generate a smart editorial title based on conversation context."""
    async with async_session_factory() as session:
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        res = await session.execute(stmt)
        conv = res.scalar_one_or_none()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        first_user_msg = next((m.content for m in conv.messages if m.role == "user"), None)
        prompt_seed = first_user_msg or conv.title or "New Chat"
        title = await generate_chat_title(prompt_seed, model=conv.model_id)

        conv.title = title
        conv.updated_at = datetime.now(timezone.utc)
        await session.commit()
        return GenerateTitleResponse(conversation_id=conv.id, title=title)


@router.patch("/conversations/{conversation_id}/messages/{message_id}", response_model=MessageResponse)
async def edit_message(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    payload: MessageUpdateRequest,
):
    """Edit the content of a previous message in PostgreSQL."""
    async with async_session_factory() as session:
        stmt = select(Message).where(
            Message.id == message_id,
            Message.conversation_id == conversation_id,
        )
        res = await session.execute(stmt)
        msg = res.scalar_one_or_none()
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found.")

        msg.content = payload.content.strip()
        await session.commit()
        await session.refresh(msg)
        return MessageResponse.model_validate(msg)


@router.post("/conversations/{conversation_id}/truncate")
async def truncate_messages(
    conversation_id: uuid.UUID,
    payload: TruncateMessagesRequest,
):
    """Truncate messages in a conversation.
    
    If after_message_id is provided, deletes all messages sequentially after that message.
    If from_message_id is provided, deletes that message and all messages after it.
    """
    async with async_session_factory() as session:
        pivot_id = payload.after_message_id or payload.from_message_id
        if not pivot_id:
            raise HTTPException(status_code=400, detail="Either after_message_id or from_message_id is required.")

        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        all_messages = (await session.execute(stmt)).scalars().all()
        pivot_idx = next((i for i, m in enumerate(all_messages) if m.id == pivot_id), None)
        if pivot_idx is None:
            raise HTTPException(status_code=404, detail="Pivot message not found in this conversation.")

        if payload.after_message_id:
            to_delete = all_messages[pivot_idx + 1:]
        else:
            to_delete = all_messages[pivot_idx:]

        count = len(to_delete)
        for m in to_delete:
            await session.delete(m)

        conv_stmt = select(Conversation).where(Conversation.id == conversation_id)
        conv = (await session.execute(conv_stmt)).scalar_one_or_none()
        if conv:
            conv.updated_at = datetime.now(timezone.utc)

        await session.commit()
        return {"deleted_count": count, "conversation_id": str(conversation_id)}


# ==============================================================================
# Persistent Streaming Chat Orchestration
# ==============================================================================

async def _auto_generate_title(conv_id: uuid.UUID, prompt: str, model: str):
    """Run LangGraph title generation workflow in background and persist title."""
    try:
        title = await generate_chat_title(prompt, model=model)
        if title:
            async with async_session_factory() as session:
                stmt = select(Conversation).where(Conversation.id == conv_id)
                res = await session.execute(stmt)
                conv = res.scalar_one_or_none()
                if conv and (conv.title in ("New Chat", "New Session", "") or not conv.title):
                    conv.title = title
                    conv.updated_at = datetime.now(timezone.utc)
                    await session.commit()
    except (asyncio.CancelledError, GeneratorExit):
        return
    except Exception as exc:
        logger.error(f"Failed to auto-generate title via LangGraph: {exc}")


async def _persist_user_turn(conv_id: uuid.UUID, user_content: str, model: str, provider: str):
    """Persists conversation and incoming user message concurrently without stalling TTFT."""
    try:
        async with async_session_factory() as session:
            stmt = select(Conversation).where(Conversation.id == conv_id)
            res = await session.execute(stmt)
            conv = res.scalar_one_or_none()
            if not conv:
                conv = Conversation(
                    id=conv_id,
                    title="New Chat",
                    model_id=model,
                    provider=provider,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(conv)

            user_msg = Message(
                id=uuid.uuid4(),
                conversation_id=conv_id,
                role="user",
                content=user_content,
                model=model,
                provider=provider,
                created_at=datetime.now(timezone.utc),
            )
            session.add(user_msg)
            await session.commit()
    except Exception as exc:
        pass


async def _persisting_stream_generator(
    conversation_id: uuid.UUID,
    user_content: str,
    model: str,
    provider: str,
    formatted_messages: List[dict],
    temperature: float,
    request: Request,
    document_ids: Optional[List[str]] = None,
    reranker_type: Optional[str] = None,
    save_user_message: bool = True,
):
    """Wraps stream_chat_workflow with fail-open L1 answer cache and PostgreSQL persistence."""
    start_time = time.perf_counter()
    accumulated_assistant_tokens: List[str] = []
    accumulated_citations: List[dict] = []
    accumulated_rag_status: Optional[dict] = None
    was_cancelled = False
    is_completed = False
    from_cache = False
    norm_q = normalize_query(user_content)
    prior_user_norms = {
        normalize_query(m.get("content", ""))
        for m in formatted_messages[:-1]
        if m.get("role") == "user" and m.get("content")
    }
    # Cache first-turn standalone queries, or exact same-thread repeat of an earlier user query.
    # Genuine follow-ups ("explain more", "what about 2024?") never match prior turns and bypass cache.
    is_cacheable = bool(norm_q) and (len(formatted_messages) == 1 or norm_q in prior_user_norms)

    try:
        # 1. Fail-open L1 exact + L2 entity-guarded semantic cache lookup (bypassed on Retry/Regenerate)
        if is_cacheable and save_user_message:
            cached = await redis_cache.get_cached_answer(
                provider=provider,
                model=model,
                reranker_type=reranker_type,
                document_ids=document_ids,
                query=user_content,
                temperature=temperature,
            )
            if (not cached or not cached.get("answer")) and settings.semantic_cache_enabled:
                cached = await redis_cache.get_semantic_cached_answer(
                    provider=provider,
                    model=model,
                    reranker_type=reranker_type,
                    document_ids=document_ids,
                    query=user_content,
                    temperature=temperature,
                    get_embedding_fn=lambda q: embeddings_client.get_embedding(q, is_query=True),
                )
            if cached and cached.get("answer"):
                from_cache = True
                cached_answer = cached["answer"]
                cached_citations = cached.get("citations") or []
                cached_rag_status = cached.get("rag_status")
                cache_tier = cached.get("cache_tier", "exact")
                cache_sim = cached.get("similarity")

                if document_ids:
                    sources_payload = {
                        "citations": cached_citations,
                        "rag_status": {**(cached_rag_status or {}), "cached": True, "cache_tier": cache_tier},
                    }
                    yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"

                meta_payload = {
                    "model": model,
                    "provider": provider,
                    "conversation_id": str(conversation_id),
                    "cached": True,
                    "cache_tier": cache_tier,
                }
                if cache_sim is not None:
                    meta_payload["similarity"] = cache_sim
                yield f"event: meta\ndata: {json.dumps(meta_payload)}\n\n"

                # Emit token deltas in small chunks so frontend streaming animations behave identically
                chunk_size = 32
                for i in range(0, len(cached_answer), chunk_size):
                    if request and await request.is_disconnected():
                        was_cancelled = True
                        return
                    delta = cached_answer[i : i + chunk_size]
                    yield f"event: token\ndata: {json.dumps({'delta': delta})}\n\n"

                total_ms = round((time.perf_counter() - start_time) * 1000, 1)
                done_payload = {
                    "status": "completed",
                    "finish_reason": "stop",
                    "cached": True,
                    "cache_tier": cache_tier,
                    "timings": {
                        "retrieval_ms": 0.0,
                        "rerank_ms": 0.0,
                        "memory_ms": 0.0,
                        "queue_wait_ms": 0.0,
                        "provider_ttft_ms": total_ms,
                        "llm_generation_ms": 0.0,
                        "llm_total_ms": 0.0,
                        "backend_total_ms": total_ms,
                        "cached": True,
                        "cache_tier": cache_tier,
                    },
                    "memory_status": {"active": False, "cached": True},
                }
                yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

                accumulated_assistant_tokens = [cached_answer]
                accumulated_citations = cached_citations
                is_completed = True
                return

        # 2. Cache miss or bypassed: stream through LangGraph immediately (zero DB wait before first token)
        stream = stream_chat_workflow(
            messages=formatted_messages,
            model=model,
            temperature=temperature,
            thread_id=str(conversation_id),
            conversation_id=str(conversation_id),
            request=request,
            document_ids=document_ids,
            reranker_type=reranker_type,
        )

        async for chunk_str in stream:
            # Check if client disconnected before yielding next chunk
            if request and await request.is_disconnected():
                logger.info("Client aborted connection; marking stream as cancelled.")
                was_cancelled = True
                break

            # Parse chunk to accumulate assistant text and detect completion
            if "event: sources" in chunk_str and "data: " in chunk_str:
                try:
                    data_part = chunk_str.split("data: ", 1)[1].strip()
                    parsed = json.loads(data_part)
                    if "citations" in parsed and isinstance(parsed["citations"], list):
                        accumulated_citations = parsed["citations"]
                    if isinstance(parsed.get("rag_status"), dict):
                        accumulated_rag_status = parsed["rag_status"]
                except Exception:
                    pass
            elif "event: token" in chunk_str and "data: " in chunk_str:
                try:
                    data_part = chunk_str.split("data: ", 1)[1].strip()
                    parsed = json.loads(data_part)
                    if "delta" in parsed:
                        accumulated_assistant_tokens.append(parsed["delta"])
                except Exception:
                    pass
            elif "event: done" in chunk_str and "data: " in chunk_str:
                try:
                    data_part = chunk_str.split("data: ", 1)[1].strip()
                    parsed = json.loads(data_part)
                    if parsed.get("status") == "completed":
                        is_completed = True
                    elif parsed.get("status") in ("cancelled", "abort"):
                        was_cancelled = True
                except Exception:
                    pass

            yield chunk_str

    except (asyncio.CancelledError, GeneratorExit):
        logger.info(f"Stream generation aborted by client for conversation {conversation_id}.")
        was_cancelled = True
        raise

    finally:
        # CRITICAL: Do NOT save incomplete assistant responses as completed messages!
        if was_cancelled or not is_completed:
            logger.info(
                f"Stream was cancelled or incomplete for conversation {conversation_id}. "
                "Discarding partial response without persisting to PostgreSQL."
            )
            return

        # ONLY save completed assistant message if generation completed naturally
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        full_response = "".join(accumulated_assistant_tokens).strip()

        if full_response:
            # Non-blocking write-behind to Redis L1 cache (0ms added to response latency)
            if is_cacheable and not from_cache and redis_cache.is_enabled:
                _schedule_cache_write(
                    redis_cache.set_cached_answer(
                        provider=provider,
                        model=model,
                        reranker_type=reranker_type,
                        document_ids=document_ids,
                        query=user_content,
                        answer=full_response,
                        citations=accumulated_citations,
                        temperature=temperature,
                        rag_status=accumulated_rag_status,
                        get_embedding_fn=(
                            (lambda q: embeddings_client.get_embedding(q, is_query=True))
                            if settings.semantic_cache_enabled
                            else None
                        ),
                    )
                )

            try:
                async with async_session_factory() as session:
                    assistant_msg = Message(
                        id=uuid.uuid4(),
                        conversation_id=conversation_id,
                        role="assistant",
                        content=full_response,
                        model=model,
                        provider=provider,
                        latency_ms=latency_ms,
                        citations=accumulated_citations,
                        created_at=datetime.now(timezone.utc),
                    )
                    session.add(assistant_msg)

                    # Update conversation record
                    stmt = select(Conversation).where(Conversation.id == conversation_id)
                    res = await session.execute(stmt)
                    conv = res.scalar_one_or_none()
                    if conv:
                        conv.updated_at = datetime.now(timezone.utc)
                        # Auto generate smart title via LangGraph if title is still default
                        if conv.title in ("New Chat", "New Session", "") and user_content:
                            asyncio.create_task(_auto_generate_title(conversation_id, user_content, model))

                    await session.commit()
            except Exception as exc:
                logger.error(f"Failed to persist completed assistant response: {exc}")


@router.post("/chat/stream")
async def chat_stream(payload: ChatStreamRequest, request: Request):
    """Stream token-by-token completion orchestrated via LangGraph, saving to PostgreSQL."""
    provider = resolve_provider(payload.model)
    conv_id = payload.conversation_id or uuid.uuid4()
    latest_user_message = payload.messages[-1].content
    formatted_messages = [{"role": m.role, "content": m.content} for m in payload.messages]

    # Concurrently ensure conversation and user message are saved without holding up the SSE stream
    # Only persist user turn if save_user_message is True (skips duplicates on retry / regenerate)
    if payload.save_user_message:
        asyncio.create_task(_persist_user_turn(conv_id, latest_user_message, payload.model, provider))

    doc_ids_str = [str(d) for d in payload.document_ids] if payload.document_ids else None

    return StreamingResponse(
        _persisting_stream_generator(
            conversation_id=conv_id,
            user_content=latest_user_message,
            model=payload.model,
            provider=provider,
            formatted_messages=formatted_messages,
            temperature=payload.temperature,
            request=request,
            document_ids=doc_ids_str,
            reranker_type=payload.reranker_type,
            save_user_message=payload.save_user_message,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
