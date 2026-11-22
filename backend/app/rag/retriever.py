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
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None

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
        self.last_dense_count: int = 0
        self.last_sparse_count: int = 0
        self.last_fused_count: int = 0

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
        """Run sparse lexical search using PostgreSQL Full-Text Search ts_rank_cd.

        Strategy:
        - Short entity/name queries (≤4 meaningful tokens): websearch_to_tsquery.
          Websearch syntax parses phrase quotes ("exact match"), explicit OR (a or b),
          and negative filters (-term), while sanitizing punctuation.
        - Long/complex queries: plainto_tsquery (strict conjunctive AND match,
          preventing query syntax errors on unformatted conversational questions).
        Note on PostgreSQL semantics: websearch_to_tsquery joins unquoted words with
        conjunction (&), not implicit disjunction (|). Disjunction only occurs when
        explicitly specified by the user with the 'or' operator.
        """
        clean_query = query_text.strip()
        if not clean_query:
            return []

        # Determine which tsquery function to use based on query length.
        # Stop words ("of", "the", "a", "is", "are", "in") are stripped by PostgreSQL's
        # English dictionary so we count meaningful tokens only.
        _STOP_WORDS = {"of", "the", "a", "an", "is", "are", "in", "on", "at", "to",
                       "and", "or", "for", "with", "by", "from", "it", "this", "that",
                       "was", "were", "be", "been", "being", "have", "has", "had",
                       "do", "does", "did", "can", "could", "will", "would", "shall",
                       "should", "may", "might", "must", "not", "no"}
        meaningful_tokens = [t for t in clean_query.lower().split() if t not in _STOP_WORDS]
        # Use websearch_to_tsquery for short entity/fact queries (≤4 meaningful tokens)
        # to support user phrases ("..."), explicit OR, and robust token handling.
        # Use plainto_tsquery for longer/complex conversational questions.
        use_websearch = len(meaningful_tokens) <= 4

        tsquery_fn = "websearch_to_tsquery" if use_websearch else "plainto_tsquery"

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
                    ts_rank_cd(to_tsvector('english', c.content), {tsquery_fn}('english', :query_text)) AS fts_score
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE to_tsvector('english', c.content) @@ {tsquery_fn}('english', :query_text)
                  {doc_filter}
                ORDER BY fts_score DESC
                LIMIT :top_k;
            """
            try:
                result = await session.execute(text(sql), params)
                rows = result.mappings().all()
                logger.debug(
                    f"FTS ({tsquery_fn}) returned {len(rows)} rows for query '{clean_query}' "
                    f"({len(meaningful_tokens)} meaningful tokens)"
                )
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
        resolved_query: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Execute parallel dense + sparse search and fuse results using Reciprocal Rank Fusion.
        
        Supports conversation-aware resolved_query and compact document candidate expansion.
        """
        start_time = time.perf_counter()

        search_query = (resolved_query or query).strip()

        # 1. Generate query embedding using configured provider (Ollama in local, Cloudflare in prod)
        query_vector = await embeddings_client.get_embedding(search_query)

        # 2. Run dense vector search and sparse full-text search in parallel
        fts_coros = [self._sparse_fts_search(query_text=search_query, document_ids=document_ids, top_k=40)]
        if resolved_query and resolved_query.strip() != query.strip():
            fts_coros.append(self._sparse_fts_search(query_text=query, document_ids=document_ids, top_k=20))

        dense_coro = self._vector_search(query_vector=query_vector, document_ids=document_ids, top_k=40)

        all_results = await asyncio.gather(dense_coro, *fts_coros)
        dense_results = all_results[0]
        sparse_results = all_results[1]
        if len(all_results) > 2:
            seen_ids = set()
            combined_sparse = []
            for r in sparse_results + all_results[2]:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    combined_sparse.append(r)
            sparse_results = combined_sparse

        self.last_dense_count = len(dense_results)
        self.last_sparse_count = len(sparse_results)

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

        # Compact document candidate expansion: for targeted small documents (<= 10 chunks),
        # ensure all chunks are present in the candidate pool so no section is accidentally missed.
        if document_ids:
            try:
                async with async_session_factory() as session:
                    all_chunks_sql = """
                        SELECT
                            c.id, c.document_id, d.filename, c.viewer_page, c.book_page,
                            c.heading, c.content, c.token_count, c.has_image, c.image_references
                        FROM document_chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE c.document_id = ANY(:doc_ids)
                        ORDER BY c.viewer_page, c.chunk_index
                        LIMIT 12;
                    """
                    doc_rows = (await session.execute(text(all_chunks_sql), {"doc_ids": [str(d) for d in document_ids]})).mappings().all()
                    if 0 < len(doc_rows) <= 10:
                        for r in doc_rows:
                            cid = r["id"]
                            if cid not in chunk_map:
                                chunk_map[cid] = {
                                    "data": dict(r),
                                    "dense_rank": None,
                                    "dense_score": 0.35,
                                    "sparse_rank": None,
                                    "sparse_score": None,
                                    "rrf": 0.005,
                                }
            except Exception as exc:
                logger.warning(f"Compact document candidate expansion skipped: {exc}")

        # Sort all chunks by fused RRF score descending
        fused_items = sorted(chunk_map.values(), key=lambda x: x["rrf"], reverse=True)
        self.last_fused_count = len(fused_items)

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
