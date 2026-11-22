"""Verification script to test and trace the active LangGraph reranker workflow at runtime."""

import asyncio
import logging
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Configure logging to capture RAG and reranker logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("reranker_verifier")

from app.config import settings
from app.ai.graph import stream_chat_workflow, create_state_graph
from app.database import async_session_factory, Document
from sqlalchemy import select


async def main():
    print("=" * 70)
    print("1. ENVIRONMENT CONFIGURATION AUDIT")
    print("=" * 70)
    print(f"RERANKER_TYPE configured:     '{settings.reranker_type}'")
    print(f"RERANKER_CANDIDATE_K:        {settings.reranker_candidate_k}")
    print(f"RERANKER_TOP_K:              {settings.reranker_top_k}")
    print(f"RERANKER_MODEL_ONNX:         '{settings.reranker_model_onnx}'")

    print("\n" + "=" * 70)
    print("2. LANGGRAPH WORKFLOW TOPOLOGY AUDIT")
    print("=" * 70)
    builder = create_state_graph()
    print("Registered Graph Nodes:", list(builder.nodes.keys()))
    print("Conditional Start Routing:")
    print("  state.document_ids truthy -> 'retrieve'")
    print("  state.document_ids empty   -> 'chat'")
    print("Execution Edges:")
    print("  START -> [route_start] -> 'retrieve' -> 'rerank' -> 'chat' -> END")

    print("\n" + "=" * 70)
    print("3. VERIFYING INGESTED DOCUMENT")
    print("=" * 70)
    doc_id = None
    async with async_session_factory() as session:
        result = await session.execute(
            select(Document).where(Document.filename == "pythonlearn.pdf").order_by(Document.created_at.desc())
        )
        doc = result.scalars().first()
        if doc:
            doc_id = str(doc.id)
            print(f"Found document: {doc.filename} (ID: {doc_id}, Chunks: {doc.chunk_count})")
        else:
            print("ERROR: pythonlearn.pdf not found in database!")
            return

    query = "How does the question mark '?' alter the greedy behavior of repetition operators in regular expressions?"
    messages = [{"role": "user", "content": query}]

    for test_reranker in [None, "onnx", "baseline"]:
        label = f"DEFAULT ({settings.reranker_type})" if test_reranker is None else test_reranker.upper()
        print("\n" + "-" * 70)
        print(f"RUNNING RUNTIME TEST WITH RERANKER: {label}")
        print("-" * 70)

        sources_received = None
        tokens_received = []

        async for sse_chunk in stream_chat_workflow(
            messages=messages,
            model="llama3.2",  # Local Ollama inference
            document_ids=[doc_id],
            reranker_type=test_reranker,
        ):
            if "event: sources" in sse_chunk:
                import json
                lines = sse_chunk.strip().split("\n")
                data_line = next((l for l in lines if l.startswith("data: ")), None)
                if data_line:
                    sources_received = json.loads(data_line[6:])
            elif "event: token" in sse_chunk:
                import json
                lines = sse_chunk.strip().split("\n")
                data_line = next((l for l in lines if l.startswith("data: ")), None)
                if data_line:
                    tok_data = json.loads(data_line[6:])
                    tokens_received.append(tok_data.get("delta", ""))
            elif "event: error" in sse_chunk:
                print(f"ERROR EVENT: {sse_chunk}")

        if sources_received:
            rag_status = sources_received.get("rag_status", {})
            citations = sources_received.get("citations", [])
            print("Captured RAG Status Payload:")
            print(f"  Retrieval:         {rag_status.get('retrieval')}")
            print(f"  Reranker:          {rag_status.get('reranker')}")
            print(f"  Candidates In:     {rag_status.get('candidates_count')}")
            print(f"  Reranked Out:      {rag_status.get('reranked_count')}")
            print(f"  Rerank Latency:    {rag_status.get('rerank_latency_ms')} ms")
            print(f"  Citations Count:   {len(citations)}")
            if citations:
                print("  Top Citation:")
                top_c = citations[0]
                print(f"    Label:   {top_c.get('citation_label')}")
                print(f"    Heading: {top_c.get('heading')}")
                print(f"    Excerpt: {top_c.get('excerpt')[:120]}...")
            print(f"  Response Generated ({len(tokens_received)} tokens):")
            sample_resp = "".join(tokens_received)[:200].replace("\n", " ")
            print(f"    \"{sample_resp}...\"")
        else:
            print("WARNING: No sources event received!")


if __name__ == "__main__":
    asyncio.run(main())
