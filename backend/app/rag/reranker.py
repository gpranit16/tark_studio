"""TARK Studio — Reranking Module.

Provides cross-encoder reranking implementations:
1. BaselineReranker: Pass-through preserving base dense + FTS + RRF ranking.
2. BGEReranker: BAAI/bge-reranker-v2-m3 via FlagEmbedding (batch scoring, CUDA/CPU).
3. ONNXReranker: Lightweight quantized cross-encoder via ONNX Runtime for Render deployment.

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


class BGEReranker(BaseReranker):
    """Local high-capacity reranker using BAAI/bge-reranker-v2-m3 via FlagEmbedding."""

    _instance: Optional["BGEReranker"] = None
    _model = None

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        if BGEReranker._model is None:
            import torch
            from FlagEmbedding import FlagReranker

            use_fp16 = torch.cuda.is_available()
            logger.info(
                f"Loading BGE reranker '{self.model_name}' (use_fp16={use_fp16}, cuda={torch.cuda.is_available()})..."
            )
            t0 = time.perf_counter()
            try:
                devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
                BGEReranker._model = FlagReranker(self.model_name, use_fp16=use_fp16, devices=devices)
                load_time = round(time.perf_counter() - t0, 3)
                logger.info(f"BGE reranker '{self.model_name}' loaded in {load_time}s.")
            except Exception as exc:
                logger.error(f"Failed to load BGE reranker '{self.model_name}': {exc}")
                raise RuntimeError(
                    f"BGE reranker initialization failed: {exc}. Refusing silent fallback."
                ) from exc

    def rerank(self, query: str, chunks: List[Any], top_k: int = 5) -> List[Any]:
        if not chunks:
            return []

        try:
            pairs = [[query, getattr(c, "content", str(c))] for c in chunks]
            scores = BGEReranker._model.compute_score(pairs)

            if isinstance(scores, (int, float)):
                scores = [scores]
            elif isinstance(scores, np.ndarray):
                scores = scores.tolist()

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
            logger.error(f"BGE reranker execution error: {exc}")
            raise RuntimeError(
                f"BGE reranker failed during inference: {exc}. Refusing silent fallback."
            ) from exc


class ONNXReranker(BaseReranker):
    """Lightweight, quantized cross-encoder running via ONNX Runtime for Render deployment."""

    _instance: Optional["ONNXReranker"] = None
    _session = None
    _tokenizer = None

    def __init__(self, model_repo: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self.model_repo = model_repo
        self._load_model()

    def _load_model(self):
        if ONNXReranker._session is None or ONNXReranker._tokenizer is None:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            from huggingface_hub import hf_hub_download

            logger.info(f"Loading ONNX quantized cross-encoder '{self.model_repo}'...")
            t0 = time.perf_counter()
            try:
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
            return []

        try:
            passages = [getattr(c, "content", str(c)) for c in chunks]
            queries = [query] * len(passages)

            inputs = ONNXReranker._tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            # Map inputs strictly to ONNX input names
            session_inputs = [inp.name for inp in ONNXReranker._session.get_inputs()]
            feed_dict = {k: v for k, v in inputs.items() if k in session_inputs}

            outputs = ONNXReranker._session.run(None, feed_dict)
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
    - 'bge': BGEReranker (BAAI/bge-reranker-v2-m3)
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
    elif reranker_type == "bge":
        settings = Settings()
        instance = BGEReranker(model_name=settings.reranker_model_bge)
    elif reranker_type == "onnx":
        settings = Settings()
        instance = ONNXReranker(model_repo=settings.reranker_model_onnx)
    else:
        raise ValueError(
            f"Unknown reranker type '{reranker_type}'. Supported: 'baseline', 'bge', 'onnx'."
        )

    _RERANKER_CACHE[reranker_type] = instance
    return instance
