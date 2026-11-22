"""Corrected Benchmark Evaluation for TARK Studio Reranker Milestone.

Enforces:
1. All 20 retrieved candidates are reranked by the cross-encoder (top_k=20).
2. Top-5 cutoff is applied only for context metrics and LLM generation.
3. HitRate@20 and Recall@20 invariance across Baseline and ONNX is mathematically verified.
4. Compares ranking quality on:
   a) Clean Production Pipeline (Dense + Strict FTS + RRF)
   b) Experimental Fallback Pipeline (Dense + OR-Fallback FTS + RRF)
5. Explains root cause of candidate pool dilution under OR-fallback.
"""

import sys
import math
import time
import json
import re
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database import async_session_factory, Document, DocumentChunk
from app.rag.retriever import hybrid_retriever
from app.rag.embeddings import embeddings_client
from app.rag.reranker import get_reranker, BaselineReranker, ONNXReranker
from eval.benchmark_dataset import BENCHMARK_QUESTIONS
from sqlalchemy import select, text


def compute_dcg(relevances: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def compute_ndcg(relevances: List[int], total_relevant: int, k: int) -> float:
    actual_dcg = compute_dcg(relevances, k)
    ideal_count = min(k, total_relevant)
    if ideal_count == 0:
        return 0.0
    ideal_dcg = compute_dcg([1] * ideal_count, k)
    return (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def is_chunk_relevant(chunk: Any, target_pages: List[str], keywords: List[str]) -> int:
    bp = str(getattr(chunk, "book_page", "") or "").lower()
    vp = str(getattr(chunk, "viewer_page", "")).lower()
    t_pages = [str(p).lower() for p in target_pages]

    if any(tp == bp or tp == vp for tp in t_pages):
        return 1

    content = getattr(chunk, "content", "")
    text_lower = content.lower()
    matches = 0
    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, text_lower):
            matches += 1
    if matches >= max(2, int(len(keywords) * 0.6)):
        return 1
    return 0


async def evaluate_pipeline(pipeline_name: str, doc_id: Any, all_doc_chunks: List[Any], use_or_fallback: bool = False):
    print("\n" + "=" * 80)
    print(f"EVALUATING PIPELINE: [{pipeline_name}] (OR Fallback: {use_or_fallback})")
    print("=" * 80)

    onnx_reranker = get_reranker("onnx")
    baseline_reranker = BaselineReranker()

    baseline_results = []
    onnx_results = []

    for item in BENCHMARK_QUESTIONS:
        qid = item["id"]
        query = item["query"]
        target_pages = item["target_pages"]
        keywords = item["keywords"]

        total_ground_truth_chunks = sum(
            1 for c in all_doc_chunks if is_chunk_relevant(c, target_pages, keywords)
        )

        # 1. Retrieve candidates
        if not use_or_fallback:
            candidates = await hybrid_retriever.retrieve(
                query=query, document_ids=[doc_id], top_k=20, max_context_tokens=4000
            )
        else:
            # Custom retrieval simulating OR-fallback FTS
            async with async_session_factory() as session:
                vec = await embeddings_client.get_embedding(query)
                d_rows = await hybrid_retriever._vector_search(vec, document_ids=[doc_id], top_k=40)
                
                # Try strict FTS
                s_rows = await hybrid_retriever._sparse_fts_search(query, document_ids=[doc_id], top_k=20)
                if not s_rows:
                    tsq_res = await session.execute(text("SELECT plainto_tsquery('english', :q)::text"), {"q": query})
                    and_tsq = tsq_res.scalar()
                    if and_tsq:
                        or_tsq = and_tsq.replace("&", "|")
                        sql_or = """
                            SELECT c.id, c.document_id, d.filename, c.viewer_page, c.book_page,
                                   c.heading, c.content, c.token_count, c.has_image, c.image_references,
                                   ts_rank_cd(to_tsvector('english', c.content), to_tsquery('english', :or_query)) AS fts_score
                            FROM document_chunks c
                            JOIN documents d ON c.document_id = d.id
                            WHERE to_tsvector('english', c.content) @@ to_tsquery('english', :or_query)
                              AND c.document_id = :doc_id
                            ORDER BY fts_score DESC
                            LIMIT 20;
                        """
                        res_or = await session.execute(text(sql_or), {"or_query": or_tsq, "doc_id": doc_id})
                        s_rows = [dict(r) for r in res_or.mappings().all()]
                
                # Fuse
                chunk_map = {}
                for rank_idx, row in enumerate(d_rows, start=1):
                    cid = row["id"]
                    chunk_map[cid] = {"data": row, "rrf": 1.0 / (60 + rank_idx)}
                for rank_idx, row in enumerate(s_rows, start=1):
                    cid = row["id"]
                    if cid not in chunk_map:
                        chunk_map[cid] = {"data": row, "rrf": 1.0 / (60 + rank_idx)}
                    else:
                        chunk_map[cid]["rrf"] += 1.0 / (60 + rank_idx)
                sorted_fused = sorted(chunk_map.values(), key=lambda x: x["rrf"], reverse=True)
                
                from app.rag.retriever import RetrievedChunk
                candidates = []
                for entry in sorted_fused[:20]:
                    r = entry["data"]
                    candidates.append(
                        RetrievedChunk(
                            chunk_id=r["id"],
                            document_id=r["document_id"],
                            document_name=r["filename"],
                            viewer_page=r["viewer_page"],
                            book_page=r.get("book_page"),
                            heading=r.get("heading"),
                            content=r["content"],
                            token_count=r.get("token_count", 0),
                            has_image=r.get("has_image", False),
                            image_references=r.get("image_references", []),
                            rrf_score=entry["rrf"],
                        )
                    )

        # Baseline: All 20 candidates in original RRF order
        baseline_candidates_20 = candidates[:20]
        # ONNX: All 20 candidates reranked by cross-encoder score
        onnx_candidates_20 = onnx_reranker.rerank(query=query, chunks=candidates[:20], top_k=20)

        # Compute query metrics
        def score_ranking(ranked_20):
            rels_20 = [is_chunk_relevant(c, target_pages, keywords) for c in ranked_20]
            hr5 = 1.0 if any(rels_20[:5]) else 0.0
            hr20 = 1.0 if any(rels_20[:20]) else 0.0
            rec5 = sum(rels_20[:5]) / total_ground_truth_chunks if total_ground_truth_chunks > 0 else 0.0
            rec20 = sum(rels_20[:20]) / total_ground_truth_chunks if total_ground_truth_chunks > 0 else 0.0
            p5 = sum(rels_20[:5]) / 5.0

            rr10 = 0.0
            for rank, r in enumerate(rels_20[:10], start=1):
                if r > 0:
                    rr10 = 1.0 / rank
                    break
            ndcg5 = compute_ndcg(rels_20, total_ground_truth_chunks, 5)

            return {
                "id": qid,
                "hit_rate_5": hr5,
                "hit_rate_20": hr20,
                "recall_5": round(rec5, 4),
                "recall_20": round(rec20, 4),
                "precision_5": round(p5, 4),
                "mrr_10": round(rr10, 4),
                "ndcg_5": round(ndcg5, 4),
            }

        b_score = score_ranking(baseline_candidates_20)
        o_score = score_ranking(onnx_candidates_20)

        # Verify mathematical invariance
        assert b_score["hit_rate_20"] == o_score["hit_rate_20"], f"HitRate@20 mismatch on Q{qid}!"
        assert b_score["recall_20"] == o_score["recall_20"], f"Recall@20 mismatch on Q{qid}!"

        baseline_results.append(b_score)
        onnx_results.append(o_score)

    def aggregate(res_list):
        n = len(res_list)
        return {
            "hit_rate_at_5": round(sum(r["hit_rate_5"] for r in res_list) / n, 4),
            "hit_rate_at_20": round(sum(r["hit_rate_20"] for r in res_list) / n, 4),
            "recall_at_5": round(sum(r["recall_5"] for r in res_list) / n, 4),
            "recall_at_20": round(sum(r["recall_20"] for r in res_list) / n, 4),
            "precision_at_5": round(sum(r["precision_5"] for r in res_list) / n, 4),
            "mrr_at_10": round(sum(r["mrr_10"] for r in res_list) / n, 4),
            "ndcg_at_5": round(sum(r["ndcg_5"] for r in res_list) / n, 4),
        }

    b_agg = aggregate(baseline_results)
    o_agg = aggregate(onnx_results)

    print(f"\nResults for {pipeline_name}:")
    print(f"{'Metric':<20} | {'Baseline (Top 20)':<20} | {'ONNX (Reranked 20)':<20} | {'Delta':<15}")
    print("-" * 80)
    for k in ["hit_rate_at_5", "hit_rate_at_20", "recall_at_5", "recall_at_20", "precision_at_5", "mrr_at_10", "ndcg_at_5"]:
        b_val = b_agg[k]
        o_val = o_agg[k]
        delta = round(o_val - b_val, 4)
        sign = "+" if delta > 0 else ""
        print(f"{k:<20} | {b_val:<20.4f} | {o_val:<20.4f} | {sign}{delta}")

    # Explicit Invariance Verification
    print("\n[Invariance Audit]:")
    print(f"  HitRate@20 Baseline == ONNX: {b_agg['hit_rate_at_20']} == {o_agg['hit_rate_at_20']} -> {b_agg['hit_rate_at_20'] == o_agg['hit_rate_at_20']}")
    print(f"  Recall@20  Baseline == ONNX: {b_agg['recall_at_20']} == {o_agg['recall_at_20']} -> {b_agg['recall_at_20'] == o_agg['recall_at_20']}")

    return {"baseline": b_agg, "onnx": o_agg, "baseline_queries": baseline_results, "onnx_queries": onnx_results}


async def main():
    async with async_session_factory() as session:
        doc = (await session.execute(
            select(Document).where(Document.filename == "pythonlearn.pdf").order_by(Document.created_at.desc())
        )).scalars().first()
        doc_id = doc.id
        chunks_res = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
        )
        all_doc_chunks = chunks_res.scalars().all()

    # Pipeline A: Production Pipeline (Strict FTS + Dense + RRF)
    prod_results = await evaluate_pipeline(
        "Production Pipeline (Strict FTS + Dense)", doc_id, all_doc_chunks, use_or_fallback=False
    )

    # Pipeline B: Experimental Pipeline (OR-Fallback FTS + Dense)
    fallback_results = await evaluate_pipeline(
        "Experimental Pipeline (OR Fallback FTS + Dense)", doc_id, all_doc_chunks, use_or_fallback=True
    )

    # Save to consolidated result file
    out_file = Path(__file__).resolve().parent / "results_corrected_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "production_pipeline": prod_results,
                "experimental_fallback_pipeline": fallback_results,
            },
            f,
            indent=2,
        )
    print(f"\nCorrected comparison saved to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
