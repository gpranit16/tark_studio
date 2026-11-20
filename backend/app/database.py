"""Database engine, models, session factory, and connectivity checking."""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy import text, Column, String, Text, Integer, Boolean, BigInteger, DateTime, ForeignKey, Index, select
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""
    pass


class Conversation(Base):
    """Conversational session / thread tracking metadata."""
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, default="New Chat")
    model_id = Column(String(100), nullable=False, default="openai/gpt-oss-120b")
    provider = Column(String(50), nullable=False, default="groq")
    is_archived = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    """Single message within a persistent conversation."""
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    model = Column(String(100), nullable=True)
    provider = Column(String(50), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    citations = Column(JSONB, nullable=True, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    conversation = relationship("Conversation", back_populates="messages")


class Document(Base):
    """Uploaded or ingested document metadata."""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)   # "pdf", "md", "txt", "py", "json"
    file_size_bytes = Column(BigInteger, nullable=False)
    page_count = Column(Integer, default=0, nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    embedding_model = Column(String(100), default="nomic-embed-text", nullable=False)
    embedding_dimension = Column(Integer, default=768, nullable=False)
    status = Column(String(50), default="processing", nullable=False)  # "processing", "completed", "failed"
    error_message = Column(Text, nullable=True)
    metadata_json = Column(JSONB, default=dict, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan", order_by="DocumentChunk.chunk_index")


class DocumentChunk(Base):
    """Text chunk derived from an ingested document with dense embedding and full-text search capability."""
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    viewer_page = Column(Integer, nullable=False)          # 1-based PDF viewer page number
    book_page = Column(String(50), nullable=True)          # Printed page number / roman label
    heading = Column(Text, nullable=True)                  # Section heading / title
    content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0, nullable=False)
    has_image = Column(Boolean, default=False, nullable=False)
    image_references = Column(JSONB, default=list, nullable=False)
    embedding = Column(Vector(768), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    document = relationship("Document", back_populates="chunks")


# Asynchronous SQLAlchemy engine
engine: AsyncEngine = create_async_engine(
    settings.async_database_url,
    pool_size=10,
    max_overflow=20,
    pool_recycle=1800,
    pool_pre_ping=True,
    echo=False,
)

# Asynchronous session factory
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def check_db_connectivity() -> Dict[str, Any]:
    """Check connectivity to PostgreSQL and verify pgvector extension availability."""
    try:
        async with engine.connect() as conn:
            res_version = await conn.execute(text("SELECT version();"))
            version_str = res_version.scalar_one_or_none()

            res_ext = await conn.execute(
                text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
            )
            ext_row = res_ext.fetchone()

            has_vector = ext_row is not None
            vector_version = ext_row[1] if has_vector else None

            return {
                "connected": True,
                "version": version_str,
                "has_vector_extension": has_vector,
                "vector_version": vector_version,
                "error": None,
            }
    except Exception as e:
        return {
            "connected": False,
            "version": None,
            "has_vector_extension": False,
            "vector_version": None,
            "error": str(e),
        }
