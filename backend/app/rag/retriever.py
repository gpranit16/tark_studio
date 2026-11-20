"""TARK Studio — Hybrid Retrieval Engine with Reciprocal Rank Fusion (RRF).

Combines:
1. Dense Vector Search (pgvector cosine similarity)
2. Sparse Lexical Search (PostgreSQL Full-Text Search ts_rank_cd)
3. Reciprocal Rank Fusion (RRF with k=60)

No reranker is used (strict baseline hybrid retrieval).
"""

import time
import uuid
import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from sqlalchemy import text

from app.database import async_session_factory
from app.rag.embeddings import embeddings_client

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard Reciprocal Rank Fusion constant


@dataclass
class RetrievedChunk:
    """Retrieved document chunk with provenance citations and rank metrics."""
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    viewer_page: int
    book_page: Optional[str]
    heading: Optional[str]
    content: str
    token_count: int
    has_image: bool
    image_references: List[Dict[str, Any]]
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: float = 0.0

    @property
    def citation_label(self) -> str:
        """Formatted human-readable citation label."""
        if self.book_page and self.book_page != str(self.viewer_page):
            return f"{self.document_name} [Page {self.book_page} (PDF p. {self.viewer_page})]"
        return f"{self.document_name} [Page {self.viewer_page}]"


class HybridRetriever:
    """Executes dense vector search and PostgreSQL FTS in parallel and fuses with RRF."""

    def __init__(self, rrf_k: int = RRF_K):
        self.rrf_k = rrf_k

    async def _vector_search(
        self,
        query_vector: List[float],
        document_ids: Optional[List[uuid.UUID]] = None,
        top_k: int = 30,
    ) -> List[Dict[str, Any]]:
        """Run dense cosine similarity vector search on document_chunks."""
        async with async_session_factory() as session:
            vector_str = "[" + ",".join(str(f) for f in query_vector) + "]"

            if document_ids:
                doc_filter = "AND c.document_id = ANY(:doc_ids)"
                params = {"query_vec": vector_str, "doc_ids": [str(d) for d in document_ids], "top_k": top_k}
            else:
                doc_filter = ""
                params = {"query_vec": vector_str, "top_k": top_k}

            sql = f"""
                SELECT
                    c.id, c.document_id, d.filename, c.viewer_page, c.book_page,
                    c.heading, c.content, c.token_count, c.has_image, c.image_references,
                    1 - (c.embedding <=> CAST(:query_vec AS vector)) AS similarity
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE 1=1 {doc_filter}
                ORDER BY c.embedding <=> CAST(:query_vec AS vector)
                LIMIT :top_k;
            """
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    async def _sparse_fts_search(
        self,
        query_text: str,
        document_ids: Optional[List[uuid.UUID]] = None,
        top_k: int = 30,
    ) -> List[Dict[str, Any]]:
        """Run sparse lexical search using PostgreSQL Full-Text Search ts_rank_cd."""
        clean_query = query_text.strip()
        if not clean_query:
            return []

        async with async_session_factory() as session:
            if document_ids:
                doc_filter = "AND c.document_id = ANY(:doc_ids)"
                params = {"query_text": clean_query, "doc_ids": [str(d) for d in document_ids], "top_k": top_k}
            else:
                doc_filter = ""
                params = {"query_text": clean_query, "top_k": top_k}

            sql = f"""
                SELECT
                    c.id, c.document_id, d.filename, c.viewer_page, c.book_page,
                    c.heading, c.content, c.token_count, c.has_image, c.image_references,
                    ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', :query_text)) AS fts_score
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', :query_text)
                  {doc_filter}
                ORDER BY fts_score DESC
                LIMIT :top_k;
            """
            try:
                result = await session.execute(text(sql), params)
                rows = result.mappings().all()
                return [dict(r) for r in rows]
            except Exception as e:
                logger.warning(f"PostgreSQL FTS query fell back for query '{clean_query}': {e}")
                return []

    async def retrieve(
        self,
        query: str,
        document_ids: Optional[List[uuid.UUID]] = None,
        top_k: int = 5,
        max_context_tokens: int = 3500,
    ) -> List[RetrievedChunk]:
        """Execute parallel dense + sparse search and fuse results using Reciprocal Rank Fusion."""
        start_time = time.perf_counter()

        # 1. Generate query embedding
        query_vector = await embeddings_client.get_embedding(query)

        # 2. Run vector search and full-text search in parallel
        dense_results, sparse_results = await asyncio.gather(
            self._vector_search(query_vector=query_vector, document_ids=document_ids, top_k=40),
            self._sparse_fts_search(query_text=query, document_ids=document_ids, top_k=40),
        )

        # 3. Reciprocal Rank Fusion calculation
        chunk_map: Dict[uuid.UUID, Dict[str, Any]] = {}

        # Process dense ranks
        for rank_idx, row in enumerate(dense_results, start=1):
            c_id = row["id"]
            if c_id not in chunk_map:
                chunk_map[c_id] = {
                    "data": row,
                    "dense_rank": rank_idx,
                    "dense_score": float(row.get("similarity", 0.0)),
                    "sparse_rank": None,
                    "sparse_score": None,
                    "rrf": 1.0 / (self.rrf_k + rank_idx),
                }
            else:
                chunk_map[c_id]["dense_rank"] = rank_idx
                chunk_map[c_id]["dense_score"] = float(row.get("similarity", 0.0))
                chunk_map[c_id]["rrf"] += 1.0 / (self.rrf_k + rank_idx)

        # Process sparse ranks
        for rank_idx, row in enumerate(sparse_results, start=1):
            c_id = row["id"]
            if c_id not in chunk_map:
                chunk_map[c_id] = {
                    "data": row,
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": rank_idx,
                    "sparse_score": float(row.get("fts_score", 0.0)),
                    "rrf": 1.0 / (self.rrf_k + rank_idx),
                }
            else:
                chunk_map[c_id]["sparse_rank"] = rank_idx
                chunk_map[c_id]["sparse_score"] = float(row.get("fts_score", 0.0))
                chunk_map[c_id]["rrf"] += 1.0 / (self.rrf_k + rank_idx)

        # Sort all chunks by fused RRF score descending
        fused_items = sorted(chunk_map.values(), key=lambda x: x["rrf"], reverse=True)

        # 4. Deduplicate and pack within token limit
        retrieved_chunks: List[RetrievedChunk] = []
        accumulated_tokens = 0

        for item in fused_items:
            row = item["data"]
            t_count = row.get("token_count", 0)

            # Check token budget
            if accumulated_tokens + t_count > max_context_tokens and len(retrieved_chunks) >= top_k:
                break

            chunk_obj = RetrievedChunk(
                chunk_id=row["id"],
                document_id=row["document_id"],
                document_name=row["filename"],
                viewer_page=row["viewer_page"],
                book_page=row["book_page"],
                heading=row["heading"],
                content=row["content"],
                token_count=t_count,
                has_image=row["has_image"],
                image_references=row.get("image_references") or [],
                dense_rank=item["dense_rank"],
                sparse_rank=item["sparse_rank"],
                dense_score=item["dense_score"],
                sparse_score=item["sparse_score"],
                rrf_score=item["rrf"],
            )
            retrieved_chunks.append(chunk_obj)
            accumulated_tokens += t_count

            if len(retrieved_chunks) >= top_k * 2 and accumulated_tokens >= max_context_tokens:
                break

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            f"Hybrid retrieval finished in {duration_ms}ms: "
            f"Dense={len(dense_results)}, Sparse={len(sparse_results)}, Fused={len(retrieved_chunks)}"
        )
        return retrieved_chunks[: max(top_k, 5)]


hybrid_retriever = HybridRetriever()
