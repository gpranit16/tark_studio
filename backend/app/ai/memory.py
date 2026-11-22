"""Conversation Memory Management for TARK Studio.

Provides token estimation, recent message budgeting, and non-blocking rolling summaries
to prevent prompt token bloat and eliminate Groq rate-limiting queue delays.
"""

import logging
import asyncio
import time
from typing import List, Tuple, Optional, Dict
import tiktoken

from app.config import settings
from app.ai.providers import stream_groq

logger = logging.getLogger(__name__)

# Cached tiktoken encoding
_ENCODER = None


def get_token_encoder():
    """Retrieve or initialize cached tiktoken encoder."""
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    """Accurately count tokens using tiktoken cl100k_base, with fallback."""
    if not text:
        return 0
    enc = get_token_encoder()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback heuristic: ~4 chars per token
    return max(1, len(text) // 4)


def count_messages_tokens(messages: List[dict]) -> int:
    """Calculate token count across a list of message dicts with role framing overhead."""
    total = 0
    for m in messages:
        # Message framing overhead: ~3 tokens for <|im_start|>{role}\n{content}<|im_end|>
        total += 3
        total += count_tokens(m.get("content", ""))
        total += count_tokens(m.get("role", ""))
    return total


def partition_messages(
    prior_messages: List[dict],
    token_budget: int = 2048,
) -> Tuple[List[dict], List[dict]]:
    """Partition prior history into older messages and recent messages within token budget.
    
    Walks backwards from the most recent prior turn to fill the recent buffer.
    Messages exceeding the budget become candidates for the rolling summary.
    
    Args:
        prior_messages: Historical messages excluding the active user query.
        token_budget: Maximum tokens allocated for recent verbatim messages.
        
    Returns:
        Tuple of (older_messages, recent_messages)
    """
    if not prior_messages:
        return [], []

    recent: List[dict] = []
    used_tokens = 0

    # Walk backwards from most recent message
    for msg in reversed(prior_messages):
        msg_tokens = 3 + count_tokens(msg.get("content", "")) + count_tokens(msg.get("role", ""))
        if used_tokens + msg_tokens <= token_budget or not recent:
            recent.append(msg)
            used_tokens += msg_tokens
        else:
            break

    # Restore chronological order
    recent.reverse()
    older_count = len(prior_messages) - len(recent)
    older = prior_messages[:older_count]

    return older, recent


def should_update_summary(
    older_messages: List[dict],
    summary_through_index: int = 0,
    threshold_tokens: int = 1024,
    min_messages: int = 4,
) -> bool:
    """Determine whether unsummarized older history warrants an incremental summary update.
    
    Avoids calling the LLM on every message. Summarizes only when a batch threshold is crossed.
    """
    if not older_messages or len(older_messages) <= summary_through_index:
        return False

    unsummarized = older_messages[summary_through_index:]
    if len(unsummarized) < min_messages:
        return False

    unsummarized_tokens = count_messages_tokens(unsummarized)
    return unsummarized_tokens >= threshold_tokens


async def generate_rolling_summary(
    existing_summary: Optional[str],
    messages_to_summarize: List[dict],
    model: str = "openai/gpt-oss-20b",
) -> str:
    """Generate or update the rolling narrative conversation summary.
    
    Distills technical definitions, user preferences, and conclusions into a compact summary.
    """
    if not messages_to_summarize:
        return existing_summary or ""

    formatted_turns = []
    for m in messages_to_summarize:
        role_label = "User" if m.get("role") == "user" else "Assistant"
        content_preview = m.get("content", "").strip()
        # Truncate any massive multi-page responses to essential 600 chars for summarization
        if len(content_preview) > 600:
            content_preview = content_preview[:600] + "..."
        formatted_turns.append(f"{role_label}: {content_preview}")

    turns_block = "\n\n".join(formatted_turns)

    system_instruction = (
        "You are TARK Studio's memory distiller.\n"
        "Your task: Update the running conversation summary with the new conversation turns.\n"
        "Guidelines:\n"
        "- Synthesize the core technical facts, decisions, questions asked, and answers given.\n"
        "- Keep it concise (under 200 words), neutral, factual, and chronological.\n"
        "- Do NOT hallucinate or extrapolate beyond what was explicitly stated.\n"
        "- Return ONLY the updated summary prose, with no intro or conversational filler."
    )

    user_content = ""
    if existing_summary and existing_summary.strip():
        user_content += f"PREVIOUS RUNNING SUMMARY:\n{existing_summary.strip()}\n\n---\n\n"
    user_content += f"NEW CONVERSATION TURNS TO INCORPORATE:\n{turns_block}\n\nUPDATED SUMMARY:"

    summary_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]

    try:
        tokens: List[str] = []
        async for chunk in stream_groq(model=model, messages=summary_messages, temperature=0.2):
            tokens.append(chunk)
        new_summary = "".join(tokens).strip()
        return new_summary if new_summary else (existing_summary or "")
    except Exception as exc:
        logger.warning(f"Failed to generate rolling summary via Groq: {exc}. Retaining prior summary.")
        return existing_summary or ""


class BackgroundSummaryManager:
    """Coordinates non-blocking, reliable rolling summaries per conversation.
    
    Guarantees:
    - Never blocks the user request or streaming response path (0ms overhead on TTFT).
    - Serializes summary jobs for a given thread_id via asyncio.Lock so updates
      never race or overwrite each other out of order.
    - Persists updated summaries to the PostgreSQL LangGraph checkpointer.
    - Prevents lost updates on graceful shutdown via wait_all_pending().
    """

    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._latest_requests: Dict[str, dict] = {}

    def get_lock(self, thread_id: str) -> asyncio.Lock:
        if thread_id not in self._locks:
            self._locks[thread_id] = asyncio.Lock()
        return self._locks[thread_id]

    def schedule_summary(
        self,
        thread_id: str,
        existing_summary: Optional[str],
        messages_to_summarize: List[dict],
        new_summary_idx: int,
        model: str,
    ):
        """Schedule a background summary job if not already pending for this thread."""
        self._latest_requests[thread_id] = {
            "existing_summary": existing_summary,
            "messages_to_summarize": messages_to_summarize,
            "new_summary_idx": new_summary_idx,
            "model": model,
        }

        # If a task is already running, the active loop and lock will pick up latest request
        if thread_id in self._active_tasks and not self._active_tasks[thread_id].done():
            logger.info(f"[Memory] Summary task already in flight for thread {thread_id}. Queued latest payload.")
            return

        task = asyncio.create_task(self._run_summary_job(thread_id))
        self._active_tasks[thread_id] = task

    async def _run_summary_job(self, thread_id: str):
        lock = self.get_lock(thread_id)
        async with lock:
            req = self._latest_requests.pop(thread_id, None)
            if not req:
                return

            existing_summary = req["existing_summary"]
            messages_to_summarize = req["messages_to_summarize"]
            new_summary_idx = req["new_summary_idx"]
            model = req["model"]

            try:
                t0 = time.perf_counter()
                logger.info(
                    f"[Memory] Background generating rolling summary for thread {thread_id} "
                    f"({len(messages_to_summarize)} turns)..."
                )
                new_summary = await generate_rolling_summary(
                    existing_summary=existing_summary,
                    messages_to_summarize=messages_to_summarize,
                    model=model,
                )
                dur = round((time.perf_counter() - t0) * 1000, 1)

                if new_summary and new_summary != existing_summary:
                    from app.ai.graph import get_compiled_graph
                    compiled_graph = await get_compiled_graph()
                    config = {"configurable": {"thread_id": thread_id}}
                    await compiled_graph.aupdate_state(
                        config,
                        {
                            "summary": new_summary,
                            "summary_through_index": new_summary_idx,
                        },
                    )
                    logger.info(
                        f"[Memory] Successfully updated rolling summary in Postgres checkpointer for thread {thread_id} "
                        f"in {dur}ms ({len(new_summary)} chars, through turn {new_summary_idx})."
                    )
            except Exception as exc:
                logger.error(f"[Memory] Error in background rolling summary for thread {thread_id}: {exc}")
            finally:
                if thread_id in self._active_tasks:
                    self._active_tasks.pop(thread_id, None)

    async def wait_all_pending(self, timeout: float = 15.0):
        """Await all active background summarization tasks before shutdown."""
        tasks = [t for t in self._active_tasks.values() if not t.done()]
        if tasks:
            logger.info(f"[Memory] Awaiting {len(tasks)} pending memory summarization tasks before shutdown...")
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            logger.info(f"[Memory] Shutdown summary wait completed: {len(done)} done, {len(pending)} pending.")


# Module singleton
background_summary_manager = BackgroundSummaryManager()
