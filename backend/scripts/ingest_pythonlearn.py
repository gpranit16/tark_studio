"""Script to ingest C:\\Users\\gupta\\Downloads\\pythonlearn.pdf into TARK Studio."""

import sys
import os
import asyncio
import time
import logging
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from app.rag.engine import ingest_document_file
from app.database import async_session_factory, Document
from sqlalchemy import select

PDF_PATH = r"C:\Users\gupta\Downloads\pythonlearn.pdf"

async def main():
    print(f"Starting ingestion of {PDF_PATH}...")
    t0 = time.time()
    try:
        doc = await ingest_document_file(PDF_PATH, filename="pythonlearn.pdf")
        elapsed = round(time.time() - t0, 2)
        print("\n=== INGESTION SUCCESS ===")
        print(f"Document ID: {doc.id}")
        print(f"Filename: {doc.filename}")
        print(f"File Type: {doc.file_type}")
        print(f"File Size: {doc.file_size_bytes} bytes ({round(doc.file_size_bytes / 1024 / 1024, 2)} MB)")
        print(f"Pages: {doc.page_count}")
        print(f"Chunks: {doc.chunk_count}")
        print(f"Embedding Model: {doc.embedding_model} ({doc.embedding_dimension}d)")
        print(f"Ingestion Duration: {elapsed} seconds")
    except Exception as e:
        print(f"\n=== INGESTION FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
