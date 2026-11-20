"""Synthetic PDF and multi-format document fixtures for TARK Studio RAG testing."""

import os
from pathlib import Path
import fitz  # PyMuPDF

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "generated"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def create_multi_page_pdf() -> str:
    """Generate a multi-page PDF with section headings and paragraphs."""
    filepath = str(FIXTURES_DIR / "multi_page_text.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 60), "Chapter 1: Foundations of TARK Studio", fontsize=16)
    p1.insert_text((50, 90), "TARK Studio is an industrial, local-first artificial intelligence workspace.", fontsize=11)
    p1.insert_text((50, 110), "It prioritizes stateful LangGraph orchestration and PostgreSQL persistence.", fontsize=11)

    p2 = doc.new_page()
    p2.insert_text((50, 60), "Chapter 2: Hybrid Retrieval Pipeline", fontsize=16)
    p2.insert_text((50, 90), "Dense vector retrieval uses pgvector cosine distance over 768d embeddings.", fontsize=11)
    p2.insert_text((50, 110), "Full-text search ranking utilizes PostgreSQL ts_rank_cd cover density.", fontsize=11)

    p3 = doc.new_page()
    p3.insert_text((50, 60), "Chapter 3: Strict Benchmark Integrity", fontsize=16)
    p3.insert_text((50, 90), "All benchmark metrics must be measured directly without synthetic fabrication.", fontsize=11)

    doc.save(filepath)
    doc.close()
    return filepath


def create_table_pdf() -> str:
    """Generate a PDF containing tabular structured data."""
    filepath = str(FIXTURES_DIR / "table_data.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 60), "Chapter 4: Comparative Benchmarks Table", fontsize=16)
    p.insert_text((50, 90), "Metric          Dense Only     FTS Only     Hybrid RRF", fontsize=11, fontname="courier")
    p.insert_text((50, 105), "--------------------------------------------------------", fontsize=11, fontname="courier")
    p.insert_text((50, 120), "Recall@5        0.7250         0.6500       0.8500", fontsize=11, fontname="courier")
    p.insert_text((50, 135), "MRR@10          0.6120         0.5430       0.7410", fontsize=11, fontname="courier")
    p.insert_text((50, 150), "P95 Latency     142ms          35ms         158ms", fontsize=11, fontname="courier")
    doc.save(filepath)
    doc.close()
    return filepath


def create_embedded_image_pdf() -> str:
    """Generate a PDF with an embedded image/pixmap."""
    filepath = str(FIXTURES_DIR / "embedded_images.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 60), "Chapter 5: Architectural Diagrams", fontsize=16)
    p.insert_text((50, 90), "Figure 5.1 illustrates the dual-stream retrieval topology.", fontsize=11)

    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 100, 100), 0)
    rect = fitz.Rect(50, 120, 200, 220)
    p.insert_image(rect, pixmap=pix)

    p.insert_text((50, 240), "Figure 5.1: Dual-stream RRF Orchestration Diagram.", fontsize=10)
    doc.save(filepath)
    doc.close()
    return filepath


def create_scanned_page_pdf() -> str:
    """Generate a PDF simulating a scanned document page (image with <15 chars of text)."""
    filepath = str(FIXTURES_DIR / "scanned_page_mock.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Scan", fontsize=8)

    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 200, 300), 0)
    rect = fitz.Rect(50, 80, 450, 680)
    p.insert_image(rect, pixmap=pix)

    doc.save(filepath)
    doc.close()
    return filepath
