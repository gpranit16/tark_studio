"""TARK Studio — Reranking Module.

Provides cross-encoder reranking implementations:
1. BaselineReranker: Pass-through preserving base dense + FTS + RRF ranking.
2. ONNXReranker: Lightweight quantized cross-encoder via ONNX Runtime for Render deployment.

Engineering constraints:
- Load models once (singleton cache).
- Never silently fall back if a configured reranker fails.
- Support batch scoring.
"""

import logging
import time
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for passage reranking."""

    @abstractmethod
    def rerank(self, query: str, chunks: List[Any], top_k: int = 5) -> List[Any]:
        """Rerank candidate chunks according to cross-encoder score."""
        pass


class BaselineReranker(BaseReranker):
    """Pass-through reranker preserving base hybrid RRF ranking."""

    def rerank(self, query: str, chunks: List[Any], top_k: int = 5) -> List[Any]:
        for idx, chunk in enumerate(chunks[:top_k], start=1):
            if isinstance(chunk, dict):
                chunk["rerank_score"] = None
                chunk["rerank_rank"] = idx
            else:
                setattr(chunk, "rerank_score", None)
                setattr(chunk, "rerank_rank", idx)
        return chunks[:top_k]


class ONNXReranker(BaseReranker):
    """Lightweight, quantized cross-encoder running via ONNX Runtime for Render deployment."""

    _instance: Optional["ONNXReranker"] = None
    _session = None
    _tokenizer = None
    model_load_time_sec: float = 0.0

    def __init__(self, model_repo: str = "Xenova/ms-marco-MiniLM-L-6-v2", max_length: int = 384):
        self.model_repo = model_repo
        self.max_length = max_length
        self.last_tok_ms: float = 0.0
        self.last_inf_ms: float = 0.0
        self._load_model()

    def _load_model(self):
        if ONNXReranker._session is None or ONNXReranker._tokenizer is None:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            from huggingface_hub import hf_hub_download

            logger.info(f"Loading ONNX quantized cross-encoder '{self.model_repo}'...")
            t0 = time.perf_counter()
            try:
                # 1. Offline-first: attempt to load from local cache to eliminate remote network calls
                try:
                    model_file = hf_hub_download(
                        repo_id=self.model_repo, filename="onnx/model_quantized.onnx", local_files_only=True
                    )
                    ONNXReranker._tokenizer = AutoTokenizer.from_pretrained(self.model_repo, local_files_only=True)
                    logger.info("Loaded ONNX model & tokenizer directly from local cache (offline mode).")
                except Exception:
                    logger.info("Local cache miss; downloading ONNX model & tokenizer from HuggingFace Hub...")
                    model_file = hf_hub_download(
                        repo_id=self.model_repo, filename="onnx/model_quantized.onnx"
                    )
                    ONNXReranker._tokenizer = AutoTokenizer.from_pretrained(self.model_repo)

                # Configure execution providers (CPUExecutionProvider for Render free/starter)
                providers = ["CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")

                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                session_options.intra_op_num_threads = min(4, os.cpu_count() or 1)

                ONNXReranker._session = ort.InferenceSession(
                    model_file, sess_options=session_options, providers=providers
                )
                load_time = round(time.perf_counter() - t0, 3)
                ONNXReranker.model_load_time_sec = load_time
                logger.info(
                    f"ONNX cross-encoder '{self.model_repo}' loaded in {load_time}s on {providers[0]}."
                )
            except Exception as exc:
                logger.error(f"Failed to load ONNX reranker '{self.model_repo}': {exc}")
                raise RuntimeError(
                    f"ONNX reranker initialization failed: {exc}. Refusing silent fallback."
                ) from exc

    def rerank(self, query: str, chunks: List[Any], top_k: int = 5) -> List[Any]:
        if not chunks:
            self.last_tok_ms = 0.0
            self.last_inf_ms = 0.0
            return []

        try:
            passages = [getattr(c, "content", str(c)) for c in chunks]
            queries = [query] * len(passages)

            # Measure tokenization latency (dynamic padding only to longest in batch)
            t_tok0 = time.perf_counter()
            inputs = ONNXReranker._tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            self.last_tok_ms = round((time.perf_counter() - t_tok0) * 1000, 2)

            # Map inputs strictly to ONNX input names
            session_inputs = [inp.name for inp in ONNXReranker._session.get_inputs()]
            feed_dict = {k: v for k, v in inputs.items() if k in session_inputs}

            # Measure raw ONNX inference latency
            t_inf0 = time.perf_counter()
            outputs = ONNXReranker._session.run(None, feed_dict)
            self.last_inf_ms = round((time.perf_counter() - t_inf0) * 1000, 2)

            raw_scores = outputs[0]

            # Flatten scores if shape is [N, 1] or [N]
            if len(raw_scores.shape) > 1:
                raw_scores = raw_scores.squeeze(-1)
            scores = raw_scores.tolist()

            if isinstance(scores, (int, float)):
                scores = [scores]

            scored_chunks = list(zip(chunks, scores))
            # Sort descending by score
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            results = []
            for rank, (chunk, score) in enumerate(scored_chunks[:top_k], start=1):
                if isinstance(chunk, dict):
                    chunk["rerank_score"] = float(score)
                    chunk["rerank_rank"] = rank
                else:
                    setattr(chunk, "rerank_score", float(score))
                    setattr(chunk, "rerank_rank", rank)
                results.append(chunk)

            return results
        except Exception as exc:
            logger.error(f"ONNX reranker execution error: {exc}")
            raise RuntimeError(
                f"ONNX reranker failed during inference: {exc}. Refusing silent fallback."
            ) from exc


# Factory function
_RERANKER_CACHE: Dict[str, BaseReranker] = {}


def get_reranker(reranker_type: Optional[str] = None) -> BaseReranker:
    """Resolve and return cached reranker instance based on type.

    Supported types:
    - 'baseline' / None: BaselineReranker
    - 'onnx': ONNXReranker (Xenova/ms-marco-MiniLM-L-6-v2)
    """
    from app.config import Settings

    if reranker_type is None:
        settings = Settings()
        reranker_type = settings.reranker_type.lower()
    else:
        reranker_type = reranker_type.lower()

    if reranker_type in _RERANKER_CACHE:
        return _RERANKER_CACHE[reranker_type]

    if reranker_type in ("baseline", "none"):
        instance = BaselineReranker()
    elif reranker_type == "onnx":
        settings = Settings()
        instance = ONNXReranker(model_repo=settings.reranker_model_onnx)
    else:
        raise ValueError(
            f"Unknown reranker type '{reranker_type}'. Supported: 'baseline', 'onnx'."
        )

    _RERANKER_CACHE[reranker_type] = instance
    return instance
