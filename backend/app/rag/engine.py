"""TARK Studio — Ingestion Engine.

Orchestrates multi-format document parsing, token-aware chunking, batch vector embeddings,
and transactional persistence into PostgreSQL + pgvector.
"""

import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import select, delete

from app.database import async_session_factory, Document, DocumentChunk
from app.rag.parser import parse_document
from app.rag.chunker import TokenChunker
from app.rag.embeddings import embeddings_client, EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)


async def ingest_document_file(
    file_path: str,
    filename: Optional[str] = None,
    document_id: Optional[uuid.UUID] = None,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 80,
) -> Document:
    """Ingest a local document file into PostgreSQL with embeddings and FTS indexing."""
    start_time = time.perf_counter()
    doc_id = document_id or uuid.uuid4()
    target_filename = filename or file_path.split("/")[-1].split("\\")[-1]

    # 1. Parse document
    logger.info(f"Parsing document {target_filename} (id: {doc_id})...")
    parsed_doc = parse_document(file_path, filename=target_filename)

    # 2. Chunk document
    chunker = TokenChunker(chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens)
    chunks_data = chunker.chunk_document(parsed_doc)
    logger.info(f"Created {len(chunks_data)} token chunks across {parsed_doc.page_count} pages.")

    # 3. Generate 768d vector embeddings via active provider (Ollama in local, Cloudflare in prod)
    logger.info(f"Generating 768d vector embeddings using {embeddings_client.model_name}...")
    chunk_texts = [c.content for c in chunks_data]
    embeddings = await embeddings_client.get_embeddings_batch(chunk_texts, is_query=False)

    if len(embeddings) != len(chunks_data):
        raise RuntimeError(f"Embedding count mismatch: expected {len(chunks_data)}, got {len(embeddings)}")

    # 4. Save to PostgreSQL
    async with async_session_factory() as session:
        # Check if record was pre-created in 'processing' state
        stmt = select(Document).where(Document.id == doc_id)
        res = await session.execute(stmt)
        doc = res.scalar_one_or_none()

        duration_sec = round(time.perf_counter() - start_time, 2)
        meta = {
            **parsed_doc.metadata,
            "ingestion_duration_sec": duration_sec,
            "chunk_size_tokens": chunk_size_tokens,
            "overlap_tokens": overlap_tokens,
            "warnings": parsed_doc.warnings[:20],
        }

        if not doc:
            doc = Document(
                id=doc_id,
                filename=target_filename,
                file_type=parsed_doc.file_type,
                file_size_bytes=parsed_doc.file_size_bytes,
                page_count=parsed_doc.page_count,
                chunk_count=len(chunks_data),
                embedding_model=embeddings_client.model_name,
                embedding_dimension=EMBEDDING_DIM,
                status="completed",
                metadata_json=meta,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            session.add(doc)
        else:
            doc.filename = target_filename
            doc.file_type = parsed_doc.file_type
            doc.file_size_bytes = parsed_doc.file_size_bytes
            doc.page_count = parsed_doc.page_count
            doc.chunk_count = len(chunks_data)
            doc.embedding_model = embeddings_client.model_name
            doc.embedding_dimension = EMBEDDING_DIM
            doc.status = "completed"
            doc.metadata_json = meta
            doc.updated_at = datetime.now(timezone.utc)

        # Delete any existing chunks for this document before adding new ones
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc_id))

        # Create chunk entities and bulk insert
        chunk_entities: List[DocumentChunk] = [
            DocumentChunk(
                id=uuid.uuid4(),
                document_id=doc_id,
                chunk_index=chunk_meta.chunk_index,
                viewer_page=chunk_meta.viewer_page,
                book_page=chunk_meta.book_page,
                heading=chunk_meta.heading,
                content=chunk_meta.content,
                token_count=chunk_meta.token_count,
                has_image=chunk_meta.has_image,
                image_references=chunk_meta.image_references,
                embedding=emb,
                created_at=datetime.now(timezone.utc),
            )
            for chunk_meta, emb in zip(chunks_data, embeddings)
        ]
        session.add_all(chunk_entities)

        await session.commit()
        await session.refresh(doc)
        logger.info(f"Ingested {target_filename} ({len(chunks_data)} chunks) in {duration_sec}s.")
        return doc
