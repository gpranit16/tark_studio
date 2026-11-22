"""TARK Studio — Token-Aware Document Chunker.

Implements token-bounded chunking with configurable overlap using tiktoken.
Preserves page coordinates, printed book labels, section headings, and figure references.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import tiktoken

from app.rag.parser import ParsedDocument, ExtractedPage

logger = logging.getLogger(__name__)

class _FallbackTokenizer:
    """Offline-safe whitespace/punctuation tokenizer approximating BPE tokens."""
    def __init__(self):
        self._pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def encode(self, text: str, disallowed_special: Any = ()) -> List[str]:
        return self._pattern.findall(text)

    def decode(self, tokens: List[str]) -> str:
        return " ".join(tokens)


_tokenizer = None


def get_tokenizer():
    """Lazily load tiktoken cl100k_base tokenizer with graceful offline fallback."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken cl100k_base ({e}); using offline fallback tokenizer.")
            _tokenizer = _FallbackTokenizer()
    return _tokenizer


def count_tokens(text: str) -> int:
    """Return exact token count for text."""
    enc = get_tokenizer()
    return len(enc.encode(text, disallowed_special=()))


@dataclass
class DocumentChunkData:
    """Single token-bounded chunk with complete provenance coordinates."""
    chunk_index: int
    viewer_page: int
    book_page: str
    heading: Optional[str]
    content: str
    token_count: int
    has_image: bool
    image_references: List[Dict[str, Any]]
    links: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)


class TokenChunker:
    """Splits extracted document pages into token-bounded chunks with overlap."""

    def __init__(
        self,
        chunk_size_tokens: int = 500,
        overlap_tokens: int = 80,
    ):
        self.chunk_size = chunk_size_tokens
        self.overlap = min(overlap_tokens, chunk_size_tokens // 2)
        self.tokenizer = get_tokenizer()

    def chunk_document(self, parsed_doc: ParsedDocument) -> List[DocumentChunkData]:
        """Convert a parsed document's pages into token-bounded chunks."""
        chunks: List[DocumentChunkData] = []
        global_chunk_idx = 0

        for page in parsed_doc.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            page_tokens = self.tokenizer.encode(page_text, disallowed_special=())
            total_tokens = len(page_tokens)

            # If page fits in a single chunk, preserve it cleanly
            sanitized_heading = page.heading.replace("\x00", "") if page.heading else None
            sanitized_text = page_text.replace("\x00", "")

            if total_tokens <= self.chunk_size:
                chunks.append(
                    DocumentChunkData(
                        chunk_index=global_chunk_idx,
                        viewer_page=page.viewer_page,
                        book_page=page.book_page,
                        heading=sanitized_heading,
                        content=sanitized_text,
                        token_count=total_tokens,
                        has_image=page.has_image,
                        image_references=page.image_references,
                        links=getattr(page, "links", []),
                        tables=getattr(page, "tables", []),
                    )
                )
                global_chunk_idx += 1
                continue

            # Slide over tokens with overlap
            step = self.chunk_size - self.overlap
            start = 0

            while start < total_tokens:
                end = min(start + self.chunk_size, total_tokens)
                slice_tokens = page_tokens[start:end]
                slice_text = self.tokenizer.decode(slice_tokens).replace("\x00", "").strip()

                if slice_text:
                    chunks.append(
                        DocumentChunkData(
                            chunk_index=global_chunk_idx,
                            viewer_page=page.viewer_page,
                            book_page=page.book_page,
                            heading=page.heading,
                            content=slice_text,
                            token_count=len(slice_tokens),
                            has_image=page.has_image,
                            image_references=page.image_references if start == 0 else [],
                            links=getattr(page, "links", []),
                            tables=getattr(page, "tables", []),
                        )
                    )
                    global_chunk_idx += 1

                if end == total_tokens:
                    break
                start += step

        return chunks
