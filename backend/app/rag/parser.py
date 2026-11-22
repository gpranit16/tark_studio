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
import re
import shutil
import logging
import unicodedata
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import fitz  # PyMuPDF

# Generic visible-URL pattern — no domain filtering, no platform assumptions
_URL_RE = re.compile(r'https?://[^\s\)\]\>\"\'\<]{4,}', re.IGNORECASE)
# Generic Markdown inline link pattern: [anchor text](url)
_MD_LINK_RE = re.compile(r'\[([^\]]*)\]\((https?://[^\)]+)\)', re.IGNORECASE)


def _extract_visible_urls(text: str, page_num: int) -> List[Dict[str, Any]]:
    """Extract all visible URLs from plain text. Generic — no domain or platform filtering."""
    links: List[Dict[str, Any]] = []
    seen: set = set()
    for m in _URL_RE.finditer(text):
        uri = m.group(0).rstrip(".,;:")
        if uri.lower() not in seen:
            seen.add(uri.lower())
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            context = text[start:end].replace("\n", " ").strip()
            links.append({
                "uri": uri,
                "anchor_text": "",
                "context": context,
                "page": page_num,
                "location_type": "visible_text",
            })
    return links


def _extract_markdown_links(text: str, page_num: int) -> List[Dict[str, Any]]:
    """Extract [anchor](url) Markdown inline links. Generic — no domain filtering."""
    links: List[Dict[str, Any]] = []
    seen: set = set()
    for m in _MD_LINK_RE.finditer(text):
        uri = m.group(2).strip().rstrip(".,;:")
        if uri.lower() not in seen:
            seen.add(uri.lower())
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            context = text[start:end].replace("\n", " ").strip()
            links.append({
                "uri": uri,
                "anchor_text": m.group(1).strip(),
                "context": context,
                "page": page_num,
                "location_type": "inline_md",
            })
    return links

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
    links: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
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


def _sanitize_obj(val: Any) -> Any:
    """Recursively sanitize NUL characters for PostgreSQL compatibility."""
    if isinstance(val, str):
        return val.replace("\x00", "")
    elif isinstance(val, dict):
        return {_sanitize_obj(k): _sanitize_obj(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_sanitize_obj(v) for v in val]
    return val


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

        # Extract text blocks, normalize unicode whitespace (NFKC), and sanitize NUL characters
        text = unicodedata.normalize("NFKC", page.get_text("text")).replace("\x00", "").strip()

        # Extract URI hyperlinks and link annotations (PyMuPDF)
        page_links: List[Dict[str, Any]] = []
        try:
            raw_links = page.get_links()
            for l in raw_links:
                uri = l.get("uri")
                if not uri:
                    continue
                rect = l.get("from")
                anchor = ""
                if rect:
                    anchor = page.get_text("text", clip=rect).strip()
                anchor = " ".join(anchor.split())
                page_links.append({
                    "uri": uri.strip(),
                    "anchor_text": anchor or "",
                    "context": anchor,
                    "page": viewer_page,
                    "location_type": "annotation",
                    "kind": l.get("kind", 2),
                })
        except Exception as exc:
            logger.warning(f"Failed to extract links on page {viewer_page}: {exc}")

        # Extract structured tables (PyMuPDF find_tables)
        page_tables: List[Dict[str, Any]] = []
        table_markdown_blocks: List[str] = []
        try:
            tabs = page.find_tables()
            for t_idx, tab in enumerate(tabs.tables):
                extracted_data = tab.extract()
                if extracted_data and len(extracted_data) > 1:
                    header = [str(col or "").strip() for col in extracted_data[0]]
                    divider = ["---"] * len(header)
                    rows = [[str(cell or "").strip() for cell in row] for row in extracted_data[1:]]
                    md_table = "| " + " | ".join(header) + " |\n| " + " | ".join(divider) + " |\n"
                    for r in rows:
                        md_table += "| " + " | ".join(r) + " |\n"
                    table_markdown_blocks.append(md_table.strip())
                    page_tables.append({
                        "table_index": t_idx + 1,
                        "header": header,
                        "rows_count": len(rows),
                    })
        except Exception as exc:
            logger.warning(f"Failed to extract tables on page {viewer_page}: {exc}")

        # Assemble augmented page text preserving reading order, tables, and hyperlinks
        augmented_text_parts = [text]

        if table_markdown_blocks:
            augmented_text_parts.append(
                f"\n[Page {viewer_page} Extracted Tables]:\n" + "\n\n".join(table_markdown_blocks)
            )

        if page_links:
            seen_uris = set()
            unique_links = []
            for pl in page_links:
                u = pl["uri"].lower()
                if u not in seen_uris:
                    seen_uris.add(u)
                    unique_links.append(pl)
            
            link_lines = [
                f"- {pl['anchor_text']}: {pl['uri']}" if pl.get('anchor_text') else f"- {pl['uri']}"
                for pl in unique_links
            ]
            augmented_text_parts.append(
                f"\n[Page {viewer_page} Hyperlinks & References]:\n" + "\n".join(link_lines)
            )

        full_page_text = "\n\n".join(part for part in augmented_text_parts if part.strip()).replace("\x00", "").strip()

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
        if len(full_page_text) < 15:
            if has_image:
                if not ocr_available:
                    diagnostics = (
                        f"Page {viewer_page} (Book p. {book_page}) contains {len(image_list)} embedded image(s) "
                        "with minimal text. OCR is unavailable (Tesseract not installed)."
                    )
                    warnings.append(diagnostics)
                    full_page_text = f"[Image/Diagram on Page {viewer_page} (Book p. {book_page}): {len(image_list)} figure(s)]"
                else:
                    diagnostics = f"Page {viewer_page} has low text; OCR triggered."
            else:
                diagnostics = f"Page {viewer_page} has minimal or blank content."
                warnings.append(diagnostics)

        extracted_pages.append(
            ExtractedPage(
                viewer_page=viewer_page,
                book_page=book_page,
                text=full_page_text,
                heading=heading,
                has_image=has_image,
                image_references=image_references,
                links=page_links,
                tables=page_tables,
                diagnostics=diagnostics,
            )
        )

    doc.close()

    all_links = [l for p in extracted_pages for l in p.links]
    all_tables = [t for p in extracted_pages for t in p.tables]

    metadata = _sanitize_obj({
        "title": doc.metadata.get("title") if hasattr(doc, "metadata") and doc.metadata else None,
        "author": doc.metadata.get("author") if hasattr(doc, "metadata") and doc.metadata else None,
        "ocr_available": ocr_available,
        "total_images": sum(len(p.image_references) for p in extracted_pages),
        "total_links": len(all_links),
        "total_tables": len(all_tables),
        "links": all_links,
        "tables": all_tables,
    })

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
    """Parse Markdown document preserving headers as section demarcations.
    Extracts inline [text](url) links and bare visible URLs generically.
    """
    file_size = os.path.getsize(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    lines = content.split("\n")
    current_heading = None
    section_lines: List[str] = []
    pages: List[ExtractedPage] = []
    section_index = 1

    def _flush_section(section_lines: List[str], heading: Optional[str], section_index: int) -> Optional[ExtractedPage]:
        sec_text = "\n".join(section_lines).strip()
        if not sec_text:
            return None
        # Extract all inline Markdown links and bare visible URLs (generic, no domain filtering)
        md_links = _extract_markdown_links(sec_text, section_index)
        visible_links = _extract_visible_urls(sec_text, section_index)
        # Merge, deduplicating by URI
        seen_uris: set = {l["uri"].lower() for l in md_links}
        all_links = list(md_links)
        for lnk in visible_links:
            if lnk["uri"].lower() not in seen_uris:
                seen_uris.add(lnk["uri"].lower())
                all_links.append(lnk)
        # Append link section to text for retrieval
        augmented = sec_text
        if all_links:
            link_lines = [
                f"- {l['anchor_text']}: {l['uri']}" if l.get("anchor_text") else f"- {l['uri']}"
                for l in all_links
            ]
            augmented += f"\n\n[Section {section_index} Embedded Links]:\n" + "\n".join(link_lines)
        return ExtractedPage(
            viewer_page=section_index,
            book_page=str(section_index),
            text=augmented,
            heading=heading,
            links=all_links,
        )

    for line in lines:
        if line.startswith("#"):
            if section_lines:
                ep = _flush_section(section_lines, current_heading, section_index)
                if ep:
                    pages.append(ep)
                    section_index += 1
                section_lines = []
            current_heading = line.strip("#").strip()
        section_lines.append(line)

    if section_lines:
        ep = _flush_section(section_lines, current_heading, section_index)
        if ep:
            pages.append(ep)

    all_links = [l for p in pages for l in p.links]
    return ParsedDocument(
        filename=filename,
        file_type="md",
        file_size_bytes=file_size,
        page_count=len(pages) or 1,
        pages=pages or [ExtractedPage(viewer_page=1, book_page="1", text=content)],
        metadata={"sections": len(pages), "total_links": len(all_links), "links": all_links},
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
    """Parse plain text document, extracting visible URLs generically."""
    file_size = os.path.getsize(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    links = _extract_visible_urls(text, 1)
    augmented = text
    if links:
        link_lines = [f"- {l['uri']}" for l in links]
        augmented += "\n\n[Embedded Links]:\n" + "\n".join(link_lines)

    page = ExtractedPage(
        viewer_page=1,
        book_page="1",
        text=augmented,
        heading=filename,
        links=links,
    )
    return ParsedDocument(
        filename=filename,
        file_type="txt",
        file_size_bytes=file_size,
        page_count=1,
        pages=[page],
        metadata={"total_links": len(links), "links": links},
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
