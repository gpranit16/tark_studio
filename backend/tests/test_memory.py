"""Unit tests for TARK Studio conversation memory, token budgeting, and background rolling summaries."""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from app.ai.memory import (
    count_tokens,
    count_messages_tokens,
    partition_messages,
    should_update_summary,
    generate_rolling_summary,
    BackgroundSummaryManager,
)


def test_count_tokens_tiktoken():
    text = "Hello world! This is a test of tiktoken counting."
    tok_count = count_tokens(text)
    assert tok_count > 0
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_count_messages_tokens():
    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is an interpreted programming language."},
    ]
    tok_count = count_messages_tokens(messages)
    assert tok_count > 10


def test_partition_messages_within_budget():
    prior_messages = [
        {"role": "user", "content": "Message 1 " * 50},
        {"role": "assistant", "content": "Response 1 " * 50},
        {"role": "user", "content": "Message 2 " * 50},
        {"role": "assistant", "content": "Response 2 " * 50},
        {"role": "user", "content": "Recent message"},
        {"role": "assistant", "content": "Recent response"},
    ]
    # Restrict budget so only the last turn fits into recent
    older, recent = partition_messages(prior_messages, token_budget=30)
    assert len(recent) >= 1
    assert recent[-1]["content"] == "Recent response"
    assert len(older) + len(recent) == len(prior_messages)


def test_should_update_summary():
    older_messages = [
        {"role": "user", "content": "Turn 1 " * 100},
        {"role": "assistant", "content": "Turn 2 " * 100},
        {"role": "user", "content": "Turn 3 " * 100},
        {"role": "assistant", "content": "Turn 4 " * 100},
    ]
    # When threshold is 200 tokens and 4 messages, it should return True
    assert should_update_summary(older_messages, summary_through_index=0, threshold_tokens=200, min_messages=4) is True

    # If already summarized through index 4, should return False
    assert should_update_summary(older_messages, summary_through_index=4, threshold_tokens=200, min_messages=4) is False

    # If fewer than min_messages unsummarized, should return False
    assert should_update_summary(older_messages[:3], summary_through_index=0, threshold_tokens=200, min_messages=4) is False


@pytest.mark.asyncio
async def test_background_summary_manager_lifecycle():
    manager = BackgroundSummaryManager()
    thread_id = "test-thread-lifecycle-001"

    messages = [
        {"role": "user", "content": "User asked about compiler"},
        {"role": "assistant", "content": "Assistant explained interpreter vs compiler"},
    ]

    with patch("app.ai.memory.generate_rolling_summary", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Summary: Python is an interpreted language."
        with patch("app.ai.graph.get_compiled_graph", new_callable=AsyncMock) as mock_graph_getter:
            mock_graph = AsyncMock()
            mock_graph_getter.return_value = mock_graph

            manager.schedule_summary(
                thread_id=thread_id,
                existing_summary=None,
                messages_to_summarize=messages,
                new_summary_idx=2,
                model="openai/gpt-oss-120b",
            )

            # Await all pending tasks in the manager
            await manager.wait_all_pending(timeout=5.0)

            mock_gen.assert_awaited_once()
            mock_graph.aupdate_state.assert_awaited_once_with(
                {"configurable": {"thread_id": thread_id}},
                {"summary": "Summary: Python is an interpreted language.", "summary_through_index": 2},
            )
