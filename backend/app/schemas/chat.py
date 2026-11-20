"""TARK Studio — Chat, Conversation, and Model Schemas."""

import uuid
from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class ChatMessage(BaseModel):
    """Single conversational turn message for requests."""
    role: Literal["user", "assistant", "system"]
    content: str


class ChatStreamRequest(BaseModel):
    """Payload for initiating a streaming chat completion."""
    conversation_id: Optional[uuid.UUID] = None
    messages: List[ChatMessage] = Field(..., min_length=1)
    model: str = Field(default="openai/gpt-oss-120b")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    save_user_message: bool = Field(default=True, description="Whether to persist the last user turn. Set False when retrying or regenerating.")
    document_ids: Optional[List[uuid.UUID]] = Field(default=None, description="Optional document UUIDs for RAG context grounding.")
    reranker_type: Optional[str] = Field(default=None, description="Optional reranker override: 'baseline', 'bge', 'onnx'.")


class MessageResponse(BaseModel):
    """Stored message response representation."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    model: Optional[str] = None
    provider: Optional[str] = None
    latency_ms: Optional[int] = None
    citations: Optional[List[dict]] = None
    created_at: datetime


class MessageUpdateRequest(BaseModel):
    """Payload to edit a previous message's content."""
    content: str = Field(..., min_length=1)


class TruncateMessagesRequest(BaseModel):
    """Payload to truncate conversation messages after or from a specific message."""
    after_message_id: Optional[uuid.UUID] = None
    from_message_id: Optional[uuid.UUID] = None


class GenerateTitleResponse(BaseModel):
    """Response containing an automatically generated conversation title."""
    conversation_id: uuid.UUID
    title: str


class ConversationCreateRequest(BaseModel):
    """Payload to create an empty conversation."""
    title: Optional[str] = Field(default="New Chat", max_length=255)
    model_id: Optional[str] = Field(default="openai/gpt-oss-120b")
    provider: Optional[str] = Field(default="groq")


class ConversationUpdateRequest(BaseModel):
    """Payload to rename or update conversation title."""
    title: str = Field(..., min_length=1, max_length=255)


class ConversationSummaryResponse(BaseModel):
    """Conversation metadata item for sidebar listing."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    model_id: str
    provider: str
    is_archived: bool = False
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    last_message_preview: Optional[str] = None


class ConversationDetailResponse(BaseModel):
    """Full conversation detail with all chronologically ordered messages."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    model_id: str
    provider: str
    is_archived: bool = False
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []


class ModelInfo(BaseModel):
    """Metadata describing a supported AI model."""
    id: str
    name: str
    provider: Literal["groq", "gemini", "ollama"]
    description: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response containing available models."""
    models: List[ModelInfo]
    default: str


class DocumentChunkResponse(BaseModel):
    """Chunk summary representation."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_index: int
    viewer_page: int
    book_page: Optional[str] = None
    heading: Optional[str] = None
    content: str
    token_count: int
    has_image: bool = False


class DocumentResponse(BaseModel):
    """Document summary item for lists and cards."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentDetailResponse(DocumentResponse):
    """Detailed document representation with sample chunks."""
    metadata_json: dict = {}
    chunks: List[DocumentChunkResponse] = []


class CitationItem(BaseModel):
    """Verified factual citation reference."""
    document_id: str
    document_name: str
    viewer_page: int
    book_page: Optional[str] = None
    citation_label: str
    excerpt: str
