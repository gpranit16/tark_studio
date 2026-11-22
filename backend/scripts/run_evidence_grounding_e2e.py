"""End-to-End Regression Test for Evidence Retrieval, Grounding, and Source-of-Truth Hierarchy."""
import sys
import os
import asyncio
import uuid
import json
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database import async_session_factory, Document
from app.ai.graph import stream_chat_workflow
from sqlalchemy import select

import unicodedata

async def run_turn(messages, doc_id, session_id):
    sources = []
    tokens = []
    rag_status = {}
    async for chunk_str in stream_chat_workflow(
        messages=messages,
        model="qwen/qwen3.8-27b",
        temperature=0.2,
        document_ids=[doc_id],
        reranker_type="onnx",
        thread_id=session_id,
    ):
        for line in chunk_str.split("\n"):
            if line.startswith("event: sources"):
                pass
            elif line.startswith("data: ") and not line.startswith("data: [DONE]"):
                try:
                    payload = json.loads(line[6:])
                    if "citations" in payload:
                        sources = payload.get("citations", [])
                        rag_status = payload.get("rag_status", {})
                    elif "delta" in payload:
                        tokens.append(payload.get("delta", ""))
                except Exception:
                    pass
    full_text = "".join(tokens).strip()
    return full_text, sources, rag_status

async def main():
    print("=" * 75)
    print("TARK Studio — Evidence Grounding & Source-of-Truth Multi-Turn Evaluation")
    print("=" * 75)

    async with async_session_factory() as session:
        doc = (await session.execute(select(Document).where(Document.filename.like("%PRANIT%")))).scalars().first()
        if not doc:
            print("ERROR: RESUME__PRANIT (6).pdf not found in database!")
            return
        doc_id = str(doc.id)
        doc_name = doc.filename

    print(f"Target Document: {doc_name} (ID: {doc_id})")

    session_id = f"test_grounding_{uuid.uuid4().hex[:8]}"
    messages = []

    test_queries = [
        ("whose resume is this ??", "Pranit Kumar"),
        ("what is his linkedin account ?", "linkedin.com/in/pranit-kumar"),
        ("extract linkedin url from his resume", "linkedin.com/in/pranit-kumar"),
        ("when he completed his internship?", "Jul 2026"),
        ("where is he from ?", "Bengaluru"),
        ("ig bengluru is mentioned in resume !", "Bengaluru"),
        ("then why you told madhya pradesh >", "Bengaluru"),
        ("where is he from ?", "Bengaluru"),
    ]

    for q_idx, (user_q, expected_substring) in enumerate(test_queries, start=1):
        print(f"\n--- Turn {q_idx}: '{user_q}' ---")
        messages.append({"role": "user", "content": user_q})

        answer, citations, status = await run_turn(messages, doc_id, session_id)
        messages.append({"role": "assistant", "content": answer})

        print(f"Answer: {answer[:180]}...")
        print(f"Citations Count: {len(citations)}")
        if citations:
            for c in citations[:2]:
                print(f"  * {c.get('citation_label')} - {c.get('excerpt', '')[:60]}...")
        print(f"Telemetry: Candidates in={status.get('candidates_before_rerank')}, Selected={status.get('evidence_selected_count')}, Links={status.get('structured_links_count')}")

        # Verification check
        norm_answer = unicodedata.normalize("NFKC", answer)
        has_expected = expected_substring.lower() in norm_answer.lower()
        print(f"Validation: {'[PASS]' if has_expected else '[FAIL]'} (Looked for '{expected_substring}')")
        if not has_expected:
            print(f"Full response:\n{answer}")
        
        await asyncio.sleep(1.0)

    print("\n" + "=" * 75)
    print("Testing Off-Topic Abstention on Resume:")
    print("=" * 75)
    off_turn_messages = [{"role": "user", "content": "What is the capital of France and what is its population?"}]
    off_answer, off_citations, off_status = await run_turn(off_turn_messages, doc_id, f"test_off_{uuid.uuid4().hex[:8]}")
    print(f"Off-topic Answer: {off_answer}")
    print(f"Off-topic Citations: {len(off_citations)}")
    assert len(off_citations) == 0, f"Expected 0 citations for off-topic query, got {len(off_citations)}"
    assert "not contain sufficient information" in off_answer.lower(), "Expected honest refusal message"
    print("[PASS] Clean honest refusal on off-topic query verified!")

if __name__ == "__main__":
    asyncio.run(main())
