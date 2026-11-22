# Generic Links Test Document

This document contains multiple unrelated URLs from different domains for testing
format-agnostic hyperlink extraction.

## Section with Inline Markdown Links

This section has links with explicit anchor text.

- Documentation: [Python Standard Library](https://docs.python.org/3/library/)
- Encyclopedia: [FastAPI](https://en.wikipedia.org/wiki/FastAPI)
- Research: [Read the paper](https://arxiv.org/abs/2005.11401)
- Misleading anchor: [This text says one thing](https://completely-different-domain.org/page)

## Section with Bare Visible URLs

Some tools emit raw URLs without anchor text.

Visit https://example.com/path/to/resource for more information.
Also check https://openai.com/research/gpt-4 and https://huggingface.co/models.

## Section with No Links

This section contains only plain prose and no hyperlinks whatsoever.
It is used to verify that extraction does not produce false positives when no URLs exist.
The quick brown fox jumps over the lazy dog.

## Section with Mixed Link Styles

Inline: [Wikipedia on RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)

Bare visible: https://github.com/langchain-ai/langchain

Another inline with a long URL: [Paper](https://arxiv.org/abs/1706.03762)
