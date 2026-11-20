"""TARK Studio — Reranker Benchmark & Comparison Evaluation.

Runs the exact same 40 benchmark questions against pythonlearn.pdf across:
1. Baseline (dense + FTS + RRF top-k, no reranker)
2. ONNX (Xenova/ms-marco-MiniLM-L-6-v2, 22MB INT8 quantized cross-encoder)
3. BGE (BAAI/bge-reranker-v2-m3 via FlagEmbedding)

Measures:
- HitRate@5 and HitRate@20
- True Recall@5 and Recall@20 (verifying Recall@20 invariance across candidate sets)
- Precision@5
- MRR@10
- nDCG@5
- Latency breakdown: Candidate Retrieval, Reranking, Total Retrieval Latency
- Peak RAM, Model Load Time, and Memory Growth
- Sample TTFT (Time to First Token) and Full Response Time
- Domain query validation (Regex greedy, SQLite foreign keys, Figure 1.1 text, Out-of-doc refusal)
"""

import sys
import math
import time
import json
import re
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import psutil
from sqlalchemy import select

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_reranker")

from app.database import async_session_factory, Document, DocumentChunk
from app.rag.retriever import hybrid_retriever
from app.rag.reranker import get_reranker, BaselineReranker, BGEReranker, ONNXReranker
from app.ai.providers import get_model_stream
from eval.benchmark_dataset import BENCHMARK_QUESTIONS


def get_process_memory_mb() -> float:
    """Return current process Resident Set Size (RSS) in megabytes."""
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)


def compute_dcg(relevances: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank cutoff k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def compute_ndcg(relevances: List[int], total_relevant: int, k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at cutoff k."""
    actual_dcg = compute_dcg(relevances, k)
    ideal_count = min(k, total_relevant)
    if ideal_count == 0:
        return 0.0

    ideal_relevances = [1] * ideal_count
    ideal_dcg = compute_dcg(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def is_chunk_relevant(chunk: Any, target_pages: List[str], keywords: List[str]) -> int:
    """Determine whether a chunk is relevant to a query based on ground truth."""
    bp = str(chunk.book_page).lower() if chunk.book_page else ""
    vp = str(chunk.viewer_page).lower()
    t_pages = [str(p).lower() for p in target_pages]

    # 1. Target page match (book page or viewer page)
    if any(tp == bp or tp == vp for tp in t_pages):
        return 1

    # 2. Strict keyword match with word boundaries
    text = chunk.content.lower()
    matches = 0
    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, text):
            matches += 1

    if matches >= max(2, int(len(keywords) * 0.6)):
        return 1

    return 0


async def evaluate_reranker(
    mode: str,
    doc_id: Any,
    all_doc_chunks: List[Any],
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RUNNING EVALUATION FOR MODE: [{mode.upper()}]")
    print("=" * 80)

    mem_before_load = get_process_memory_mb()
    t_load_start = time.perf_counter()

    # Instantiate / get reranker
    reranker = get_reranker(mode)
    load_time_sec = round(time.perf_counter() - t_load_start, 3)
    mem_after_load = get_process_memory_mb()
    load_mem_delta_mb = round(mem_after_load - mem_before_load, 2)

    print(f"Model Load Time: {load_time_sec}s | Memory Before: {mem_before_load}MB -> After: {mem_after_load}MB (Delta: +{load_mem_delta_mb}MB)")

    results_per_query = []
    candidate_latencies: List[float] = []
    rerank_latencies: List[float] = []
    total_latencies: List[float] = []

    hit_rate_at_5_list = []
    hit_rate_at_20_list = []
    recall_at_5_list = []
    recall_at_20_list = []
    precision_at_5_list = []
    mrr_at_10_list = []
    ndcg_at_5_list = []

    peak_ram = mem_after_load

    for item in BENCHMARK_QUESTIONS:
        qid = item["id"]
        cat = item["category"]
        query = item["query"]
        target_pages = item["target_pages"]
        keywords = item["keywords"]

        total_ground_truth_chunks = sum(
            1 for c in all_doc_chunks if is_chunk_relevant(c, target_pages, keywords)
        )

        # 1. Retrieve 20 candidates via base hybrid retrieval (identical across all modes)
        t_cand_start = time.perf_counter()
        candidates = await hybrid_retriever.retrieve(
            query=query,
            document_ids=[doc_id],
            top_k=20,
            max_context_tokens=4000,
        )
        cand_ms = round((time.perf_counter() - t_cand_start) * 1000, 2)
        candidate_latencies.append(cand_ms)

        # 2. Rerank candidates to top 5
        t_rerank_start = time.perf_counter()
        reranked = reranker.rerank(query=query, chunks=candidates, top_k=5)
        rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000, 2)
        rerank_latencies.append(rerank_ms)

        total_ms = round(cand_ms + rerank_ms, 2)
        total_latencies.append(total_ms)

        # Track memory usage during run
        current_mem = get_process_memory_mb()
        if current_mem > peak_ram:
            peak_ram = current_mem

        # Binary relevance on full candidate set (20)
        cand_rel = [is_chunk_relevant(c, target_pages, keywords) for c in candidates]
        # Binary relevance on reranked top 5
        top5_rel = [is_chunk_relevant(c, target_pages, keywords) for c in reranked]

        # 1. HitRate@K
        hr5 = 1.0 if any(top5_rel[:5]) else 0.0
        hr20 = 1.0 if any(cand_rel[:20]) else 0.0

        # 2. True Recall@K
        rec5 = (
            (sum(top5_rel[:5]) / total_ground_truth_chunks)
            if total_ground_truth_chunks > 0
            else 0.0
        )
        rec20 = (
            (sum(cand_rel[:20]) / total_ground_truth_chunks)
            if total_ground_truth_chunks > 0
            else 0.0
        )

        # 3. Precision@5
        p5 = sum(top5_rel[:5]) / 5.0

        # 4. MRR@10: Computed over the ranked output.
        # For top 5 reranked, followed by the remaining candidates in their order
        combined_ranked_rel = top5_rel + [rel for c, rel in zip(candidates, cand_rel) if c not in reranked]
        mrr = 0.0
        for rank_idx, rel in enumerate(combined_ranked_rel[:10], start=1):
            if rel == 1:
                mrr = 1.0 / rank_idx
                break

        # 5. nDCG@5
        ndcg5 = compute_ndcg(top5_rel, total_ground_truth_chunks, 5)

        hit_rate_at_5_list.append(hr5)
        hit_rate_at_20_list.append(hr20)
        recall_at_5_list.append(rec5)
        recall_at_20_list.append(rec20)
        precision_at_5_list.append(p5)
        mrr_at_10_list.append(mrr)
        ndcg_at_5_list.append(ndcg5)

        top_citation = reranked[0].citation_label if reranked else "None"
        top_score = getattr(reranked[0], "rerank_score", None) if reranked else None
        score_str = f"score={top_score:+.2f}" if top_score is not None else "score=N/A"

        print(
            f"[{qid:02d}/40] {cat[:16]:16s} | HR@5={hr5:.0f} Rec@5={rec5:.2f} P@5={p5:.2f} MRR={mrr:.2f} nDCG={ndcg5:.2f} | Rerank: {rerank_ms:5.1f}ms (Tot: {total_ms:6.1f}ms) | {score_str} | Top: {top_citation}"
        )

        results_per_query.append({
            "id": qid,
            "category": cat,
            "query": query,
            "candidate_retrieval_ms": cand_ms,
            "rerank_ms": rerank_ms,
            "total_latency_ms": total_ms,
            "total_ground_truth_chunks": total_ground_truth_chunks,
            "hit_rate_5": hr5,
            "hit_rate_20": hr20,
            "recall_5": round(rec5, 4),
            "recall_20": round(rec20, 4),
            "precision_5": round(p5, 4),
            "mrr_10": round(mrr, 4),
            "ndcg_5": round(ndcg5, 4),
            "top_citation": top_citation,
            "top_rerank_score": round(top_score, 4) if top_score is not None else None,
        })

    # Summary latency stats
    mean_cand_lat = round(sum(candidate_latencies) / len(candidate_latencies), 2)
    mean_rerank_lat = round(sum(rerank_latencies) / len(rerank_latencies), 2)
    mean_tot_lat = round(sum(total_latencies) / len(total_latencies), 2)

    sorted_tot = sorted(total_latencies)
    p50_tot = round(sorted_tot[len(sorted_tot) // 2], 2)
    p95_tot = round(sorted_tot[int(math.ceil(0.95 * len(sorted_tot))) - 1], 2)

    # Measure sample TTFT and full response time
    print(f"\nMeasuring TTFT and Full Response Time for [{mode.upper()}] with active model...")
    sample_queries = [
        "What is a dictionary in Python?",
        "How do you parse XML using ElementTree in Python?",
        "Explain regular expression greedy matching with examples.",
    ]
    ttft_samples: List[float] = []
    full_time_samples: List[float] = []

    for sq in sample_queries:
        cands = await hybrid_retriever.retrieve(query=sq, document_ids=[doc_id], top_k=20)
        reranked_sample = reranker.rerank(query=sq, chunks=cands, top_k=5)
        context = "\n".join(c.content for c in reranked_sample)
        prompt_msgs = [
            {"role": "system", "content": f"Answer concisely based strictly on:\n{context[:1500]}"},
            {"role": "user", "content": sq},
        ]
        t0 = time.perf_counter()
        first_token_time = None
        stream = get_model_stream(model="qwen/qwen3.8-27b", messages=prompt_msgs, temperature=0.2)
        async for _delta in stream:
            if first_token_time is None:
                first_token_time = round((time.perf_counter() - t0) * 1000, 2)
        full_time = round((time.perf_counter() - t0) * 1000, 2)
        if first_token_time:
            ttft_samples.append(first_token_time)
        full_time_samples.append(full_time)

    mean_ttft = round(sum(ttft_samples) / len(ttft_samples), 2) if ttft_samples else 0.0
    mean_full_time = round(sum(full_time_samples) / len(full_time_samples), 2) if full_time_samples else 0.0

    summary = {
        "mode": mode,
        "reranker_name": (
            "Baseline (None)"
            if mode == "baseline"
            else ("BAAI/bge-reranker-v2-m3" if mode == "bge" else "Xenova/ms-marco-MiniLM-L-6-v2 (INT8 ONNX)")
        ),
        "total_queries": len(BENCHMARK_QUESTIONS),
        "embedding_model": "nomic-embed-text (768d)",
        "memory_profile": {
            "initial_ram_mb": mem_before_load,
            "after_load_ram_mb": mem_after_load,
            "load_delta_mb": load_mem_delta_mb,
            "peak_ram_mb": peak_ram,
            "model_load_time_sec": load_time_sec,
        },
        "latency_profile": {
            "candidate_retrieval_mean_ms": mean_cand_lat,
            "reranking_mean_ms": mean_rerank_lat,
            "total_retrieval_mean_ms": mean_tot_lat,
            "total_retrieval_p50_ms": p50_tot,
            "total_retrieval_p95_ms": p95_tot,
            "ttft_mean_ms": mean_ttft,
            "full_response_mean_ms": mean_full_time,
        },
        "metrics": {
            "hit_rate_at_5": round(sum(hit_rate_at_5_list) / len(hit_rate_at_5_list), 4),
            "hit_rate_at_20": round(sum(hit_rate_at_20_list) / len(hit_rate_at_20_list), 4),
            "recall_at_5": round(sum(recall_at_5_list) / len(recall_at_5_list), 4),
            "recall_at_20": round(sum(recall_at_20_list) / len(recall_at_20_list), 4),
            "precision_at_5": round(sum(precision_at_5_list) / len(precision_at_5_list), 4),
            "mrr_at_10": round(sum(mrr_at_10_list) / len(mrr_at_10_list), 4),
            "ndcg_at_5": round(sum(ndcg_at_5_list) / len(ndcg_at_5_list), 4),
        },
        "query_results": results_per_query,
    }

    out_file = Path(__file__).resolve().parent / f"results_{mode}.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Results for [{mode}] successfully saved to: {out_file}")

    return summary


async def run_specialized_tests(doc_id: Any):
    """Run test verification for the 4 critical RAG cases:
    1. Regex greedy vs non-greedy (`[0-9]+` vs `.*?`)
    2. SQLite foreign keys & referential integrity (Chapter 15)
    3. Hardware architecture figure understanding (Figure 1.1 / extracted text)
    4. Out-of-document queries (Honest refusal with 0 citations)
    """
    print("\n" + "=" * 80)
    print("RUNNING SPECIALIZED RAG ISSUE VERIFICATIONS")
    print("=" * 80)

    from app.ai.graph import get_compiled_graph
    from langgraph.checkpoint.memory import MemorySaver

    test_saver = MemorySaver()
    from app.ai.graph import build_chat_graph
    graph = build_chat_graph(checkpointer=test_saver)

    cases = [
        {
            "name": "Regex greedy vs non-greedy",
            "query": "In Python regular expressions, is .*? greedy or non-greedy? What about [0-9]+?",
            "expect_in_context": [".*?", "greedy"],
            "verify_off_topic": False,
        },
        {
            "name": "SQLite foreign keys / referential integrity",
            "query": "In SQLite and Chapter 15, does SQLite enforce foreign key constraints automatically when defining Track and Artist tables?",
            "expect_in_context": ["sqlite", "foreign"],
            "verify_off_topic": False,
        },
        {
            "name": "Hardware architecture diagram",
            "query": "Based on Figure 1.1 hardware architecture diagram, what components are shown?",
            "expect_in_context": ["hardware", "cpu", "memory"],
            "verify_off_topic": False,
        },
        {
            "name": "Out-of-document query (Honest refusal)",
            "query": "What is the capital city of France and what is its population?",
            "expect_in_context": [],
            "verify_off_topic": True,
        },
    ]

    for mode in ["baseline", "onnx", "bge"]:
        print(f"\n--- Testing Specialized Cases for Mode: {mode.upper()} ---")
        for c in cases:
            session_id = f"test_{mode}_{c['name'].replace(' ', '_')}"
            config = {"configurable": {"thread_id": session_id}}

            state_input = {
                "messages": [{"role": "user", "content": c["query"]}],
                "document_ids": [str(doc_id)],
                "reranker_type": mode,
                "model": "qwen/qwen3.8-27b",
            }

            events = []
            citations = []
            response_tokens = []
            async for chunk_event in graph.astream(state_input, config=config, stream_mode="custom"):
                ev_type = chunk_event.get("type")
                if ev_type == "sources":
                    citations = chunk_event.get("data", {}).get("citations", [])
                elif ev_type == "token":
                    response_tokens.append(chunk_event.get("data", {}).get("delta", ""))

            full_resp = "".join(response_tokens)
            print(f"Case: {c['name']}")
            print(f"  Citations Count: {len(citations)}")
            print(f"  First 120 chars: {full_resp[:120].strip()}...")
            if c["verify_off_topic"]:
                # In reranker modes, off-topic queries should yield 0 citations and honest refusal
                if mode in ("bge", "onnx"):
                    assert len(citations) == 0, f"Expected 0 citations for off-topic query in {mode}, got {len(citations)}"
                    assert "do not contain sufficient information" in full_resp.lower() or "not contain" in full_resp.lower(), "Expected honest refusal message"
                    print("  [PASS] Clean honest refusal with zero citations verified!")
            else:
                assert len(citations) > 0, f"Expected citations for valid query in {mode}"
                print("  [PASS] Grounded response with citations generated!")


async def main():
    # 1. Fetch document
    async with async_session_factory() as session:
        stmt = select(Document).where(Document.filename == "pythonlearn.pdf").order_by(Document.created_at.desc())
        doc = (await session.execute(stmt)).scalars().first()
        if not doc:
            raise RuntimeError("Document pythonlearn.pdf not found in database.")
        doc_id = doc.id

        chunk_stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
        all_doc_chunks = (await session.execute(chunk_stmt)).scalars().all()

    print(f"Loaded target document: {doc.filename} (ID: {doc_id}) with {len(all_doc_chunks)} chunks.")

    # 2. Run all three evaluations sequentially
    summaries = {}
    for mode in ["baseline", "onnx", "bge"]:
        summaries[mode] = await evaluate_reranker(mode, doc_id, all_doc_chunks)

    # 3. Print Cross-Comparison Table
    print("\n" + "=" * 95)
    print("HEAD-TO-HEAD BENCHMARK COMPARISON: BASELINE vs ONNX vs BGE")
    print("=" * 95)
    header = f"{'Metric':<25} | {'Baseline':<12} | {'ONNX (MiniLM)':<15} | {'BGE (v2-m3)':<15} | {'Delta (BGE - Base)':<18}"
    print(header)
    print("-" * 95)

    metric_keys = [
        ("HitRate@5", "hit_rate_at_5", True),
        ("HitRate@20", "hit_rate_at_20", True),
        ("True Recall@5", "recall_at_5", True),
        ("True Recall@20 (Candidate)", "recall_at_20", True),
        ("Precision@5", "precision_at_5", True),
        ("MRR@10", "mrr_at_10", False),
        ("nDCG@5", "ndcg_at_5", False),
    ]

    for label, key, is_pct in metric_keys:
        v_base = summaries["baseline"]["metrics"][key]
        v_onnx = summaries["onnx"]["metrics"][key]
        v_bge = summaries["bge"]["metrics"][key]
        delta_bge = v_bge - v_base

        if is_pct:
            s_base = f"{v_base * 100:.2f}%"
            s_onnx = f"{v_onnx * 100:.2f}%"
            s_bge = f"{v_bge * 100:.2f}%"
            s_delta = f"{delta_bge * 100:+.2f}% pts"
        else:
            s_base = f"{v_base:.4f}"
            s_onnx = f"{v_onnx:.4f}"
            s_bge = f"{v_bge:.4f}"
            s_delta = f"{delta_bge:+.4f}"

        print(f"{label:<25} | {s_base:<12} | {s_onnx:<15} | {s_bge:<15} | {s_delta:<18}")

    print("-" * 95)
    print("LATENCY & MEMORY PROFILE:")
    print("-" * 95)
    print(f"{'Candidate Retrieval (Mean)':<25} | {summaries['baseline']['latency_profile']['candidate_retrieval_mean_ms']:<8.1f} ms  | {summaries['onnx']['latency_profile']['candidate_retrieval_mean_ms']:<11.1f} ms  | {summaries['bge']['latency_profile']['candidate_retrieval_mean_ms']:<11.1f} ms")
    print(f"{'Rerank Latency (Mean)':<25} | {'0.0':<8} ms  | {summaries['onnx']['latency_profile']['reranking_mean_ms']:<11.1f} ms  | {summaries['bge']['latency_profile']['reranking_mean_ms']:<11.1f} ms")
    print(f"{'Total Retrieval (Mean)':<25} | {summaries['baseline']['latency_profile']['total_retrieval_mean_ms']:<8.1f} ms  | {summaries['onnx']['latency_profile']['total_retrieval_mean_ms']:<11.1f} ms  | {summaries['bge']['latency_profile']['total_retrieval_mean_ms']:<11.1f} ms")
    print(f"{'Model Load Time':<25} | {'0.00':<8} s   | {summaries['onnx']['memory_profile']['model_load_time_sec']:<11.2f} s   | {summaries['bge']['memory_profile']['model_load_time_sec']:<11.2f} s")
    print(f"{'RAM Growth from Model':<25} | {'0.0':<8} MB  | {summaries['onnx']['memory_profile']['load_delta_mb']:<11.1f} MB  | {summaries['bge']['memory_profile']['load_delta_mb']:<11.1f} MB")
    print(f"{'Peak Process RAM':<25} | {summaries['baseline']['memory_profile']['peak_ram_mb']:<8.1f} MB  | {summaries['onnx']['memory_profile']['peak_ram_mb']:<11.1f} MB  | {summaries['bge']['memory_profile']['peak_ram_mb']:<11.1f} MB")
    print(f"{'Time To First Token (TTFT)':<25} | {summaries['baseline']['latency_profile']['ttft_mean_ms']:<8.1f} ms  | {summaries['onnx']['latency_profile']['ttft_mean_ms']:<11.1f} ms  | {summaries['bge']['latency_profile']['ttft_mean_ms']:<11.1f} ms")
    print("=" * 95)

    # 4. Verify Recall@20 Invariance Principle
    r20_base = summaries["baseline"]["metrics"]["recall_at_20"]
    r20_onnx = summaries["onnx"]["metrics"]["recall_at_20"]
    r20_bge = summaries["bge"]["metrics"]["recall_at_20"]
    print(f"\n[Verification] Recall@20 across runs: Baseline={r20_base}, ONNX={r20_onnx}, BGE={r20_bge}")
    assert r20_base == r20_onnx == r20_bge, "CRITICAL ERROR: Recall@20 must be identical across all runs!"
    print("[Verification PASSED] Recall@20 is mathematically identical across all 3 runs.")

    # 5. Run specialized domain tests
    await run_specialized_tests(doc_id)


if __name__ == "__main__":
    asyncio.run(main())
