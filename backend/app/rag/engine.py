"""TARK Studio — Ingestion Engine.

Orchestrates multi-format document parsing, token-aware chunking, batch vector embeddings,
and transactional persistence into PostgreSQL + pgvector.
"""

import time
import uuid
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import select, delete

from app.database import async_session_factory, Document, DocumentChunk
from app.rag.parser import parse_document, ParsedDocument
from app.rag.chunker import TokenChunker
from app.rag.embeddings import embeddings_client, EMBEDDING_MODEL, EMBEDDING_DIM
from app.rag.vision import vision_client

logger = logging.getLogger(__name__)


async def _resolve_scanned_pages_vision(file_path: str, parsed_doc: ParsedDocument) -> bool:
    """Render scanned candidate pages to 150 DPI JPEG and transcribe concurrently via Cloudflare Vision.
    
    Returns True if any candidate page failed to transcribe (leading to partial status).
    """
    import fitz

    tasks = []
    candidate_pages = []
    doc = None
    try:
        doc = fitz.open(file_path)
        for page in parsed_doc.pages:
            if getattr(page, "needs_vision", False):
                page_idx = page.viewer_page - 1
                if 0 <= page_idx < len(doc):
                    pdf_page = doc[page_idx]
                    pix = pdf_page.get_pixmap(dpi=150)
                    jpeg_bytes = pix.tobytes(output="jpeg", jpg_quality=85)
                    tasks.append(vision_client.transcribe_image_bytes(jpeg_bytes))
                    candidate_pages.append(page)
    except Exception as exc:
        logger.warning(f"Failed to open document for vision snapshot: {exc}")
        return True
    finally:
        if doc is not None:
            doc.close()

    if not tasks:
        return False

    # Execute concurrent vision inference bounded across candidate pages
    results = await asyncio.gather(*tasks, return_exceptions=True)
    any_failures = False

    for page, res in zip(candidate_pages, results):
        if isinstance(res, str) and res.strip():
            page.text = res.strip()
            page.provenance = "vision_transcription"
            page.needs_vision = False
            page.diagnostics = f"Page {page.viewer_page}: Transcribed via Cloudflare Vision OCR ({len(res)} chars)."
            logger.info(f"Page {page.viewer_page} successfully transcribed via Cloudflare Vision ({len(res)} chars).")
        else:
            any_failures = True
            page.provenance = "scanned_untranscribed"
            page.needs_vision = False
            err_detail = str(res) if isinstance(res, Exception) else "Vision OCR timed out or empty"
            warning_msg = f"Page {page.viewer_page}: Vision transcription unavailable ({err_detail})."
            parsed_doc.warnings.append(warning_msg)
            page.text = f"[Scanned Page {page.viewer_page}: Visual reference only]"

    return any_failures


async def ingest_document_file(
    file_path: str,
    filename: Optional[str] = None,
    document_id: Optional[uuid.UUID] = None,
    chunk_size_tokens: int = 400,
    overlap_tokens: int = 60,
) -> Document:
    """Ingest a local document file into PostgreSQL with embeddings and FTS indexing."""
    start_time = time.perf_counter()
    doc_id = document_id or uuid.uuid4()
    target_filename = filename or file_path.split("/")[-1].split("\\")[-1]

    # 1. Parse document
    logger.info(f"Parsing document {target_filename} (id: {doc_id})...")
    t0 = time.perf_counter()
    parsed_doc = parse_document(file_path, filename=target_filename)
    parse_sec = round(time.perf_counter() - t0, 2)

    # 1b. Asynchronously resolve true scanned pages via Cloudflare Workers AI Vision
    has_partial_failures = False
    vision_candidates = [p for p in parsed_doc.pages if getattr(p, "needs_vision", False)]
    if vision_candidates and vision_client.is_configured:
        logger.info(f"Resolving {len(vision_candidates)} scanned candidate pages via Cloudflare Vision OCR...")
        t_vis = time.perf_counter()
        has_partial_failures = await _resolve_scanned_pages_vision(file_path, parsed_doc)
        logger.info(f"Cloudflare Vision OCR pass finished in {round(time.perf_counter() - t_vis, 2)}s.")

    # 2. Chunk document
    t0 = time.perf_counter()
    chunker = TokenChunker(chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens)
    chunks_data = chunker.chunk_document(parsed_doc)
    chunk_sec = round(time.perf_counter() - t0, 2)
    logger.info(f"Created {len(chunks_data)} token chunks across {parsed_doc.page_count} pages.")

    # 3. Generate 768d vector embeddings via active provider (Ollama in local, Cloudflare in prod)
    logger.info(f"Generating 768d vector embeddings using {embeddings_client.model_name}...")
    t0 = time.perf_counter()
    chunk_texts = [c.content for c in chunks_data]
    embeddings = await embeddings_client.get_embeddings_batch(chunk_texts, is_query=False)
    embed_sec = round(time.perf_counter() - t0, 2)

    if len(embeddings) != len(chunks_data):
        raise RuntimeError(f"Embedding count mismatch: expected {len(chunks_data)}, got {len(embeddings)}")

    # 4. Save to PostgreSQL
    t0 = time.perf_counter()
    async with async_session_factory() as session:
        # Check if record was pre-created in 'processing' state
        stmt = select(Document).where(Document.id == doc_id)
        res = await session.execute(stmt)
        doc = res.scalar_one_or_none()

        duration_sec = round(time.perf_counter() - start_time, 2)
        target_status = "partial" if has_partial_failures else "completed"
        meta = {
            **parsed_doc.metadata,
            "ingestion_duration_sec": duration_sec,
            "parse_duration_sec": parse_sec,
            "chunk_duration_sec": chunk_sec,
            "embed_duration_sec": embed_sec,
            "chunk_size_tokens": chunk_size_tokens,
            "overlap_tokens": overlap_tokens,
            "warnings": parsed_doc.warnings[:20],
            "page_provenances": [getattr(p, "provenance", "native_digital") for p in parsed_doc.pages],
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
                status=target_status,
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
            doc.status = target_status
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
                image_references=[{"provenance": getattr(chunk_meta, "provenance", "native_digital")}] + list(chunk_meta.image_references),
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
