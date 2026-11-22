"""TARK Studio — Document Management & Ingestion API Endpoints.

Provides endpoints to:
- Upload documents (PDF, MD, TXT, PY, JSON)
- List documents with processing status
- Inspect document details & chunk previews
- Delete documents and cascading vectors
"""

import os
import uuid
import shutil
import asyncio
import logging
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload

from app.database import async_session_factory, Document, DocumentChunk
from app.schemas.chat import (
    DocumentResponse,
    DocumentDetailResponse,
    DocumentChunkResponse,
)
from app.rag.parser import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from app.rag.engine import ingest_document_file
from app.rag.embeddings import embeddings_client, EMBEDDING_DIM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Document Intelligence & RAG"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Strong reference set to prevent Python garbage collection from terminating active ingestion tasks
_active_ingestion_tasks: set[asyncio.Task] = set()


def _launch_ingestion_task(file_path: str, filename: str, doc_id: uuid.UUID) -> asyncio.Task:
    """Launch background ingestion coroutine and retain a strong reference until completion."""
    task = asyncio.create_task(_run_background_ingestion(file_path, filename, doc_id))
    _active_ingestion_tasks.add(task)
    task.add_done_callback(_active_ingestion_tasks.discard)
    return task


async def _run_background_ingestion(file_path: str, filename: str, doc_id: uuid.UUID):
    """Run ingestion in background task, updating status and error on failure."""
    try:
        await ingest_document_file(file_path=file_path, filename=filename, document_id=doc_id)
    except Exception as exc:
        logger.exception(f"Background ingestion failed for {filename} ({doc_id}): {exc}")
        async with async_session_factory() as session:
            stmt = select(Document).where(Document.id == doc_id)
            res = await session.execute(stmt)
            doc = res.scalar_one_or_none()
            if doc:
                doc.status = "failed"
                doc.error_message = str(exc)
                doc.updated_at = datetime.now(timezone.utc)
                await session.commit()


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a document file (PDF, MD, TXT, PY, JSON) for vectorization & full-text search indexing."""
    filename = file.filename or "uploaded_document"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    doc_id = uuid.uuid4()
    saved_filename = f"{doc_id}_{filename}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    # Stream write to disk and check file size limit
    bytes_written = 0
    with open(saved_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            bytes_written += len(chunk)
            if bytes_written > MAX_FILE_SIZE_BYTES:
                buffer.close()
                if os.path.exists(saved_path):
                    os.remove(saved_path)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum allowable size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.",
                )
            buffer.write(chunk)

    # Create initial document record in 'processing' state
    async with async_session_factory() as session:
        doc = Document(
            id=doc_id,
            filename=filename,
            file_type=ext.lstrip("."),
            file_size_bytes=bytes_written,
            page_count=0,
            chunk_count=0,
            embedding_model=embeddings_client.model_name,
            embedding_dimension=EMBEDDING_DIM,
            status="processing",
            metadata_json={"stored_path": saved_path},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)

    # Kick off asynchronous background ingestion with strong reference retention
    _launch_ingestion_task(saved_path, filename, doc_id)

    return DocumentResponse.model_validate(doc)


@router.get("", response_model=List[DocumentResponse])
async def list_documents():
    """List all ingested and processing documents ordered by created_at descending."""
    async with async_session_factory() as session:
        stmt = select(Document).order_by(desc(Document.created_at))
        result = await session.execute(stmt)
        docs = result.scalars().all()

        # Check and fail any stale processing documents older than 5 minutes
        now = datetime.now(timezone.utc)
        has_stale = False
        for d in docs:
            if d.status == "processing" and (now - d.created_at).total_seconds() > 300:
                d.status = "failed"
                d.error_message = "Ingestion timed out (process was interrupted)"
                d.updated_at = now
                has_stale = True

        if has_stale:
            await session.commit()

        return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: uuid.UUID):
    """Retrieve detailed document metadata and sample chunk previews."""
    async with async_session_factory() as session:
        stmt = (
            select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.id == document_id)
        )
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

        # Take first 15 chunks as preview
        sample_chunks = [
            DocumentChunkResponse.model_validate(c)
            for c in sorted(doc.chunks, key=lambda x: x.chunk_index)[:15]
        ]

        return DocumentDetailResponse(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size_bytes=doc.file_size_bytes,
            page_count=doc.page_count,
            chunk_count=doc.chunk_count,
            embedding_model=doc.embedding_model,
            embedding_dimension=doc.embedding_dimension,
            status=doc.status,
            error_message=doc.error_message,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            metadata_json=doc.metadata_json or {},
            chunks=sample_chunks,
        )


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(document_id: uuid.UUID):
    """Delete a document and all cascading chunk vectors from PostgreSQL."""
    async with async_session_factory() as session:
        stmt = select(Document).where(Document.id == document_id)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

        # Clean file from disk if stored locally
        stored_path = doc.metadata_json.get("stored_path") if doc.metadata_json else None
        if stored_path and os.path.exists(stored_path):
            try:
                os.remove(stored_path)
            except Exception as e:
                logger.warning(f"Failed to remove file from disk: {e}")

        await session.delete(doc)
        await session.commit()
        return {"deleted": True, "id": str(document_id)}
