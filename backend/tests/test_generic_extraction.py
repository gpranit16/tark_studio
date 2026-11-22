"""Generic document extraction tests — format-agnostic, domain-agnostic.

These tests verify that the extraction pipeline:
- Works on Markdown and plain text without domain-specific assumptions.
- Produces a unified link schema: uri, anchor_text, page, location_type.
- Does not filter or suppress URLs based on domain or platform.
- Returns empty link lists for documents with no URLs.
- Preserves link URIs in assembled chunk text for retrieval.
"""

import os
import pytest
from pathlib import Path

from app.rag.parser import parse_markdown, parse_plain_text, parse_document

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GENERIC_MD = FIXTURES_DIR / "generic_links_test.md"
NO_LINKS_TXT = FIXTURES_DIR / "no_links.txt"

# ---------------------------------------------------------------------------
# Unified schema helpers
# ---------------------------------------------------------------------------

REQUIRED_LINK_KEYS = {"uri", "anchor_text", "page", "location_type"}


def _assert_link_schema(link: dict) -> None:
    """Every link dict must carry the unified schema keys."""
    missing = REQUIRED_LINK_KEYS - link.keys()
    assert not missing, f"Link missing keys {missing}: {link}"
    assert link["uri"].startswith("http"), f"URI should be an http(s) URL: {link['uri']}"
    assert isinstance(link["page"], int), f"page must be int: {link}"
    assert link["location_type"] in ("annotation", "inline_md", "visible_text"), (
        f"Unknown location_type: {link['location_type']}"
    )


# ---------------------------------------------------------------------------
# 1. Markdown inline link extraction
# ---------------------------------------------------------------------------

def test_markdown_inline_link_extraction():
    """[text](url) links must be extracted from Markdown sections."""
    doc = parse_markdown(str(GENERIC_MD), GENERIC_MD.name)
    all_links = [l for page in doc.pages for l in page.links]
    inline_links = [l for l in all_links if l["location_type"] == "inline_md"]

    assert len(inline_links) >= 3, f"Expected ≥3 inline Markdown links, got {len(inline_links)}"

    uris = [l["uri"] for l in inline_links]
    assert any("docs.python.org" in u for u in uris), "docs.python.org link missing"
    assert any("arxiv.org" in u for u in uris), "arxiv.org link missing"
    assert any("completely-different-domain.org" in u for u in uris), (
        "Misleading-anchor link must still be extracted"
    )


# ---------------------------------------------------------------------------
# 2. Markdown bare visible URL extraction
# ---------------------------------------------------------------------------

def test_markdown_visible_url_extraction():
    """Bare https:// URLs in Markdown text must also be extracted."""
    doc = parse_markdown(str(GENERIC_MD), GENERIC_MD.name)
    all_links = [l for page in doc.pages for l in page.links]
    visible = [l for l in all_links if l["location_type"] == "visible_text"]

    uris = [l["uri"] for l in visible]
    assert any("example.com" in u for u in uris), "example.com bare URL missing"
    assert any("openai.com" in u for u in uris), "openai.com bare URL missing"
    assert any("huggingface.co" in u for u in uris), "huggingface.co bare URL missing"


# ---------------------------------------------------------------------------
# 3. Unified schema enforcement
# ---------------------------------------------------------------------------

def test_extraction_produces_unified_schema():
    """Every extracted link must conform to the unified schema."""
    doc = parse_markdown(str(GENERIC_MD), GENERIC_MD.name)
    all_links = [l for page in doc.pages for l in page.links]
    assert len(all_links) > 0, "Expected links in the generic Markdown fixture"
    for link in all_links:
        _assert_link_schema(link)


# ---------------------------------------------------------------------------
# 4. No-link document returns empty list
# ---------------------------------------------------------------------------

def test_no_links_document_returns_empty():
    """A plain text document with no URLs must have an empty links list."""
    doc = parse_plain_text(str(NO_LINKS_TXT), NO_LINKS_TXT.name)
    all_links = [l for page in doc.pages for l in page.links]
    assert all_links == [], f"Expected no links, got {all_links}"


# ---------------------------------------------------------------------------
# 5. Multiple domains — no filtering by platform
# ---------------------------------------------------------------------------

def test_multiple_domains_no_filtering():
    """All URL domains must be returned regardless of platform or category."""
    doc = parse_markdown(str(GENERIC_MD), GENERIC_MD.name)
    all_links = [l for page in doc.pages for l in page.links]
    uris = [l["uri"] for l in all_links]

    expected_domains = [
        "docs.python.org",
        "en.wikipedia.org",
        "arxiv.org",
        "completely-different-domain.org",
        "example.com",
        "openai.com",
        "huggingface.co",
        "github.com",
    ]
    for domain in expected_domains:
        assert any(domain in u for u in uris), (
            f"Domain '{domain}' was not extracted — filtering may be occurring"
        )


# ---------------------------------------------------------------------------
# 6. Link URIs appear in chunk text (retrieval signal)
# ---------------------------------------------------------------------------

def test_link_in_chunk_content():
    """Extracted link URIs must appear inside the page text for retrieval."""
    doc = parse_markdown(str(GENERIC_MD), GENERIC_MD.name)
    for page in doc.pages:
        for link in page.links:
            assert link["uri"] in page.text, (
                f"URI '{link['uri']}' not found in page text — retrieval will miss it"
            )


# ---------------------------------------------------------------------------
# 7. Plain text visible URL extraction
# ---------------------------------------------------------------------------

def test_plain_text_visible_url_extraction(tmp_path):
    """Plain text files with visible URLs must have them extracted."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text(
        "Check https://example.org/page and https://docs.pytest.org/en/latest/ for details.\n"
        "No filter: https://totally-unknown-domain.io/resource"
    )
    doc = parse_plain_text(str(txt_file), txt_file.name)
    all_links = [l for page in doc.pages for l in page.links]
    uris = [l["uri"] for l in all_links]

    assert any("example.org" in u for u in uris), "example.org missing"
    assert any("docs.pytest.org" in u for u in uris), "docs.pytest.org missing"
    assert any("totally-unknown-domain.io" in u for u in uris), "unknown domain filtered out — should not be"
    for link in all_links:
        _assert_link_schema(link)


# ---------------------------------------------------------------------------
# 8. parse_document() dispatches correctly
# ---------------------------------------------------------------------------

def test_parse_document_dispatches_to_markdown():
    """parse_document must dispatch .md files to the Markdown parser."""
    doc = parse_document(str(GENERIC_MD), GENERIC_MD.name)
    assert doc.file_type == "md"
    all_links = [l for page in doc.pages for l in page.links]
    assert len(all_links) > 0, "parse_document should produce links for Markdown"
