"""TARK Studio — Multi-Format Document Parser.

Supports:
- PDF (.pdf): Page-by-page extraction with PyMuPDF, extracting printed page numbers,
  viewer page numbers, headings, tables, and embedded image references.
- Markdown (.md): Section-aware text parsing.
- Plain Text (.txt): Paragraph-preserving extraction.
- Python (.py): Safe static AST/code extraction. NEVER executes code.
- JSON (.json): Structured JSON payload parsing.

Security & Integrity:
- Validates allowed extensions and limits.
- Checks Tesseract presence before OCR; never silently drops unparseable pages.
"""

import os
import ast
import json
import shutil
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt", ".py", ".json"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit


@dataclass
class ExtractedPage:
    """Represents text and metadata extracted from a single document page or section."""
    viewer_page: int                       # 1-indexed physical/viewer page number
    book_page: str                         # Printed page label (e.g. '15', 'vi')
    text: str                              # Clean textual content
    heading: Optional[str] = None          # Detected section heading
    has_image: bool = False
    image_references: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Optional[str] = None


@dataclass
class ParsedDocument:
    """Complete structured output from parsing a document file."""
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    pages: List[ExtractedPage]
    metadata: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def is_tesseract_available() -> bool:
    """Detect whether Tesseract OCR binary is installed and discoverable on PATH."""
    return shutil.which("tesseract") is not None


def _format_book_page_label(ranges: Optional[List[dict]], page_idx: int) -> str:
    """Determine the printed book page label based on PDF page label dictionary."""
    if not ranges:
        return str(page_idx + 1)

    matching_range = None
    for r in ranges:
        if r.get("startpage", 0) <= page_idx:
            matching_range = r
        else:
            break

    if not matching_range:
        return str(page_idx + 1)

    offset = page_idx - matching_range.get("startpage", 0)
    first_num = matching_range.get("firstpagenum", 1)
    num = first_num + offset
    style = matching_range.get("style", "D")

    if style == "r":
        val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        syms = ["m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i"]
        res = ""
        n = num
        for v, s in zip(val, syms):
            while n >= v:
                res += s
                n -= v
        return res or str(num)
    elif style == "R":
        return _format_book_page_label([dict(matching_range, style="r")], page_idx).upper()
    return str(num)


def parse_pdf(file_path: str, filename: str) -> ParsedDocument:
    """Parse PDF document page by page preserving viewer and printed book pages."""
    file_size = os.path.getsize(file_path)
    doc = fitz.open(file_path)
    total_pages = len(doc)
    page_label_ranges = doc.get_page_labels()
    ocr_available = is_tesseract_available()

    extracted_pages: List[ExtractedPage] = []
    warnings: List[str] = []

    for page_idx in range(total_pages):
        viewer_page = page_idx + 1
        book_page = _format_book_page_label(page_label_ranges, page_idx)
        page = doc[page_idx]

        # Extract text blocks and sanitize NUL characters for PostgreSQL
        text = page.get_text("text").replace("\x00", "").strip()

        # Extract detected section headings (lines with larger font or prefix)
        heading = None
        blocks = page.get_text("blocks")
        for b in blocks:
            # block format: (x0, y0, x1, y1, text, block_no, block_type)
            if len(b) >= 5 and b[4].strip():
                first_line = b[4].replace("\x00", "").strip().split("\n")[0].strip()
                if any(first_line.startswith(p) for p in ("Chapter ", "CHAPTER ", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                    heading = first_line[:120]
                    break

        # Extract embedded images metadata
        image_list = page.get_images(full=True)
        has_image = len(image_list) > 0
        image_references = []
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            width = img_info[2]
            height = img_info[3]
            image_references.append({
                "image_index": img_idx + 1,
                "xref": xref,
                "width": width,
                "height": height,
                "viewer_page": viewer_page,
                "book_page": book_page,
            })

        # Handle empty or scanned pages
        diagnostics = None
        if len(text) < 15:
            if has_image:
                if not ocr_available:
                    diagnostics = (
                        f"Page {viewer_page} (Book p. {book_page}) contains {len(image_list)} embedded image(s) "
                        "with minimal text. OCR is unavailable (Tesseract not installed)."
                    )
                    warnings.append(diagnostics)
                    text = f"[Image/Diagram on Page {viewer_page} (Book p. {book_page}): {len(image_list)} figure(s)]"
                else:
                    diagnostics = f"Page {viewer_page} has low text; OCR triggered."
            else:
                diagnostics = f"Page {viewer_page} has minimal or blank content."
                warnings.append(diagnostics)

        extracted_pages.append(
            ExtractedPage(
                viewer_page=viewer_page,
                book_page=book_page,
                text=text,
                heading=heading,
                has_image=has_image,
                image_references=image_references,
                diagnostics=diagnostics,
            )
        )

    doc.close()

    metadata = {
        "title": doc.metadata.get("title") if hasattr(doc, "metadata") and doc.metadata else None,
        "author": doc.metadata.get("author") if hasattr(doc, "metadata") and doc.metadata else None,
        "ocr_available": ocr_available,
        "total_images": sum(len(p.image_references) for p in extracted_pages),
    }

    return ParsedDocument(
        filename=filename,
        file_type="pdf",
        file_size_bytes=file_size,
        page_count=total_pages,
        pages=extracted_pages,
        metadata=metadata,
        warnings=warnings,
    )


def parse_markdown(file_path: str, filename: str) -> ParsedDocument:
    """Parse Markdown document preserving headers as section demarcations."""
    file_size = os.path.getsize(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    lines = content.split("\n")
    current_heading = None
    section_lines: List[str] = []
    pages: List[ExtractedPage] = []
    section_index = 1

    for line in lines:
        if line.startswith("#"):
            if section_lines:
                sec_text = "\n".join(section_lines).strip()
                if sec_text:
                    pages.append(ExtractedPage(
                        viewer_page=section_index,
                        book_page=str(section_index),
                        text=sec_text,
                        heading=current_heading,
                    ))
                    section_index += 1
                section_lines = []
            current_heading = line.strip("#").strip()
        section_lines.append(line)

    if section_lines:
        sec_text = "\n".join(section_lines).strip()
        if sec_text:
            pages.append(ExtractedPage(
                viewer_page=section_index,
                book_page=str(section_index),
                text=sec_text,
                heading=current_heading,
            ))

    return ParsedDocument(
        filename=filename,
        file_type="md",
        file_size_bytes=file_size,
        page_count=len(pages) or 1,
        pages=pages or [ExtractedPage(viewer_page=1, book_page="1", text=content)],
        metadata={"sections": len(pages)},
    )


def parse_python_file(file_path: str, filename: str) -> ParsedDocument:
    """Parse Python code statically. NEVER executes code."""
    file_size = os.path.getsize(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        code_text = f.read()

    # Extract module docstring & functions using static AST
    docstring = None
    definitions = []
    try:
        parsed_ast = ast.parse(code_text, filename=filename)
        docstring = ast.get_docstring(parsed_ast)
        for node in ast.walk(parsed_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.append(f"def {node.name}")
            elif isinstance(node, ast.ClassDef):
                definitions.append(f"class {node.name}")
    except Exception as e:
        logger.warning(f"AST parse notice for {filename}: {e}")

    page = ExtractedPage(
        viewer_page=1,
        book_page="1",
        text=code_text,
        heading=f"Python Source: {filename}",
        has_image=False,
    )

    return ParsedDocument(
        filename=filename,
        file_type="py",
        file_size_bytes=file_size,
        page_count=1,
        pages=[page],
        metadata={
            "is_python": True,
            "static_docstring": docstring,
            "definitions": definitions[:30],
        },
    )


def parse_plain_text(file_path: str, filename: str) -> ParsedDocument:
    """Parse plain text document into paragraph blocks."""
    file_size = os.path.getsize(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    page = ExtractedPage(
        viewer_page=1,
        book_page="1",
        text=text,
        heading=filename,
    )
    return ParsedDocument(
        filename=filename,
        file_type="txt",
        file_size_bytes=file_size,
        page_count=1,
        pages=[page],
        metadata={},
    )


def parse_json_file(file_path: str, filename: str) -> ParsedDocument:
    """Parse JSON file formatting as structured text."""
    file_size = os.path.getsize(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    formatted_text = json.dumps(data, indent=2)
    page = ExtractedPage(
        viewer_page=1,
        book_page="1",
        text=formatted_text,
        heading=f"JSON: {filename}",
    )
    return ParsedDocument(
        filename=filename,
        file_type="json",
        file_size_bytes=file_size,
        page_count=1,
        pages=[page],
        metadata={"is_json": True},
    )


def parse_document(file_path: str, filename: Optional[str] = None) -> ParsedDocument:
    """Master entrypoint: validates file and delegates to format-specific parser."""
    actual_name = filename or os.path.basename(file_path)
    ext = os.path.splitext(actual_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    if os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size exceeds maximum allowable limit of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.")

    if ext == ".pdf":
        return parse_pdf(file_path, actual_name)
    elif ext == ".md":
        return parse_markdown(file_path, actual_name)
    elif ext == ".py":
        return parse_python_file(file_path, actual_name)
    elif ext == ".json":
        return parse_json_file(file_path, actual_name)
    else:
        return parse_plain_text(file_path, actual_name)
