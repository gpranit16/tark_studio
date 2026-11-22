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


def _sync_ocr_single_page(file_path: str, page_idx: int) -> str:
    """Run PyMuPDF C++ OCR on a single composite page off the main event loop in a worker thread.
    
    Uses full=True at 72 DPI to render the complete composite page (combining all scan layers,
    stamps, and signatures) into a lightweight 500k-pixel canvas for ultra-low latency (<2s).
    """
    import fitz
    doc = None
    try:
        doc = fitz.open(file_path)
        if 0 <= page_idx < len(doc):
            page = doc[page_idx]

            # 1. Fast composite page OCR at 72 DPI (~595x842 = 0.5MP, sub-2s inference)
            try:
                tp = page.get_textpage_ocr(language="eng", dpi=72, full=True)
                ocr_text = page.get_text(textpage=tp).strip()
                if len(ocr_text) >= 15:
                    return ocr_text
            except Exception as exc:
                logger.debug(f"72 DPI OCR pass on page {page_idx + 1}: {exc}")

            # 2. High-resolution fallback at 96 DPI for dense or small fonts
            try:
                tp = page.get_textpage_ocr(language="eng", dpi=96, full=True)
                ocr_text = page.get_text(textpage=tp).strip()
                if len(ocr_text) >= 15:
                    return ocr_text
            except Exception as exc:
                logger.debug(f"96 DPI OCR fallback on page {page_idx + 1}: {exc}")
    except Exception as exc:
        logger.warning(f"PyMuPDF OCR worker error on page {page_idx + 1}: {exc}")
    finally:
        if doc is not None:
            doc.close()
    return ""


def _sync_render_grayscale_jpeg(file_path: str, page_idx: int) -> Optional[bytes]:
    """Render a 72 DPI grayscale JPEG for secondary vision fallback, checking for blank pages."""
    import fitz
    doc = None
    try:
        doc = fitz.open(file_path)
        if 0 <= page_idx < len(doc):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=72, colorspace=fitz.csGRAY)
            samples = memoryview(pix.samples)
            if min(samples) > 240:
                # Blank / uniform unprinted page
                return None
            return pix.tobytes(output="jpeg", jpg_quality=70)
    except Exception as exc:
        logger.warning(f"Failed to render page pixmap on page {page_idx + 1}: {exc}")
        return None
    finally:
        if doc is not None:
            doc.close()
    return None


async def _resolve_scanned_pages_vision(file_path: str, parsed_doc: ParsedDocument) -> bool:
    """Transcribe scanned candidate pages concurrently using bounded worker threads and Vision fallback.
    
    1. Primary: Local PyMuPDF C++ OCR offloaded to background threads (asyncio.to_thread).
    2. Instant skip: Blank/uniform pages detected in <1ms.
    3. Secondary: Cloudflare Workers AI Llama-3.2 Vision fallback bounded by concurrency semaphore.
    """
    candidate_pages = [p for p in parsed_doc.pages if getattr(p, "needs_vision", False)]
    if not candidate_pages:
        return False

    sem = asyncio.Semaphore(3)
    any_failures = False

    async def _process_page(page):
        nonlocal any_failures
        page_idx = page.viewer_page - 1

        # Step 1: Ultra-fast Local In-Container OCR in worker thread (non-blocking)
        try:
            local_text = await asyncio.to_thread(_sync_ocr_single_page, file_path, page_idx)
            if local_text and len(local_text) >= 15:
                page.text = local_text
                page.provenance = "ocr_layer"
                page.needs_vision = False
                page.diagnostics = f"Page {page.viewer_page}: Local C++ OCR transcribed ({len(local_text)} chars)."
                logger.info(f"Page {page.viewer_page} transcribed via local C++ OCR ({len(local_text)} chars).")
                return
        except Exception as exc:
            logger.warning(f"Local OCR failed on page {page.viewer_page}: {exc}")

        # Step 2: Render 96 DPI grayscale JPEG in worker thread
        jpeg_bytes = await asyncio.to_thread(_sync_render_grayscale_jpeg, file_path, page_idx)
        if not jpeg_bytes:
            # Blank or uniform page
            page.text = ""
            page.provenance = "native_digital"
            page.needs_vision = False
            page.diagnostics = f"Page {page.viewer_page}: Blank unprinted page."
            return

        # Step 3: Secondary fallback to Cloudflare Workers AI Vision
        if vision_client.account_id and vision_client.api_token:
            async with sem:
                try:
                    cf_text = await vision_client.transcribe_image_bytes(jpeg_bytes)
                    if cf_text and cf_text.strip():
                        page.text = cf_text.strip()
                        page.provenance = "vision_transcription"
                        page.needs_vision = False
                        page.diagnostics = f"Page {page.viewer_page}: Transcribed via Vision OCR ({len(cf_text)} chars)."
                        logger.info(f"Page {page.viewer_page} transcribed via Cloudflare Vision ({len(cf_text)} chars).")
                        return
                except Exception as exc:
                    logger.warning(f"Vision transcription failed for page {page.viewer_page}: {exc}")

        # Step 4: Graceful fallback without crashing
        any_failures = True
        page.provenance = "scanned_untranscribed"
        page.needs_vision = False
        warning_msg = f"Page {page.viewer_page}: Vision transcription unavailable."
        parsed_doc.warnings.append(warning_msg)
        page.text = f"[Scanned Page {page.viewer_page}: Visual reference only]"

    # Run candidate pages concurrently with bounded semaphore off the main event loop
    await asyncio.gather(*[_process_page(p) for p in candidate_pages], return_exceptions=True)
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

    # 1b. Asynchronously resolve true scanned pages via ultra-fast local OCR & Vision pipeline
    has_partial_failures = False
    vision_candidates = [p for p in parsed_doc.pages if getattr(p, "needs_vision", False)]
    if vision_candidates:
        logger.info(f"Resolving {len(vision_candidates)} scanned candidate pages via fast OCR & Vision pipeline...")
        t_vis = time.perf_counter()
        has_partial_failures = await _resolve_scanned_pages_vision(file_path, parsed_doc)
        logger.info(f"Vision & OCR pass finished in {round(time.perf_counter() - t_vis, 2)}s.")

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
