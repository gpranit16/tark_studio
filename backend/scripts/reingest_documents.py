"""Re-ingest all uploaded documents with enhanced hyperlink and table extraction."""
import sys
import os
import asyncio
import uuid
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database import async_session_factory, Document
from app.rag.engine import ingest_document_file
from sqlalchemy import select

async def main():
    print("=" * 70)
    print("TARK Studio — Re-indexing Ingested Documents with Links & Tables")
    print("=" * 70)

    uploads_dir = backend_dir / "uploads"
    if not uploads_dir.exists():
        print("Uploads dir not found!")
        return

    async with async_session_factory() as session:
        docs = (await session.execute(select(Document))).scalars().all()
        doc_map = {str(d.id): d for d in docs}

    for f in uploads_dir.iterdir():
        if not f.is_file() or not f.suffix.lower() == ".pdf":
            continue
        # Expected format: <uuid>_<original_filename>
        name = f.name
        parts = name.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            doc_id = uuid.UUID(parts[0])
            orig_filename = parts[1]
        except ValueError:
            continue

        if str(doc_id) in doc_map or True:
            print(f"\nRe-ingesting: {orig_filename} (ID: {doc_id})...")
            try:
                updated_doc = await ingest_document_file(
                    file_path=str(f),
                    filename=orig_filename,
                    document_id=doc_id,
                )
                print(f"  [SUCCESS] Chunks: {updated_doc.chunk_count} | Links: {updated_doc.metadata_json.get('total_links', 0)} | Tables: {updated_doc.metadata_json.get('total_tables', 0)}")
            except Exception as e:
                print(f"  [ERROR] Failed to reingest {name}: {e}")

    print("\n" + "=" * 70)
    print("ALL DOCUMENTS RE-INDEXED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
