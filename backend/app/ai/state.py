"""LangGraph state definitions for TARK Studio AI workflows."""

from typing import TypedDict, List, Optional, Any


class ChatState(TypedDict, total=False):
    """Core state for chat workflows in LangGraph.
    
    Attributes:
        messages: List of chat message dictionaries with 'role' and 'content'.
        model: Identifier of the target model (e.g. 'openai/gpt-oss-120b').
        temperature: Model sampling temperature between 0.0 and 2.0.
        provider: Resolved provider name ('groq', 'gemini', or 'ollama').
        response: Accumulated assistant completion text.
        error: Error description if a node fails, else None.
    """
    messages: List[dict]
    model: str
    temperature: float
    provider: str
    response: str
    error: Optional[str]
    document_ids: Optional[List[str]]
    candidate_chunks: Optional[List[Any]]
    retrieved_chunks: Optional[List[dict]]
    citations: Optional[List[dict]]
    reranker_type: Optional[str]
    retrieval_latency_ms: Optional[float]
    rerank_latency_ms: Optional[float]
    tok_ms: Optional[float]
    inf_ms: Optional[float]
    resolved_query: Optional[str]
    is_followup: Optional[bool]
    is_correction: Optional[bool]
    dense_candidates_count: Optional[int]
    sparse_candidates_count: Optional[int]
    fused_candidates_count: Optional[int]
    # Agentic web search fields (only active when document_ids is empty in Cloud mode)
    web_search_mode: Optional[str]
    web_search_used: Optional[bool]
    intent_latency_ms: Optional[float]
    web_search_latency_ms: Optional[float]
    # Conversation memory fields
    thread_id: Optional[str]
    summary: Optional[str]
    summary_through_index: Optional[int]
    recent_messages: Optional[List[dict]]
    estimated_prompt_tokens: Optional[int]
    memory_overhead_ms: Optional[float]
    memory_summary_status: Optional[str]
    memory_recent_tokens: Optional[int]
    memory_summary_turns: Optional[int]


class TitleState(TypedDict, total=False):
    """State definition for LangGraph title generation workflow."""
    prompt: str
    model: str
    provider: str
    title: str
    error: Optional[str]
