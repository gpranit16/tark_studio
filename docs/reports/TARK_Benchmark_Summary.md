# TARK Studio — RAG Performance Benchmark Summary

**Audited & Verified:** November 23, 2026  
**Hardware Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)  
**Model Stack:** Meta Llama 3.2 3B (Quantized Q4_K_M) + BAAI/bge-reranker-base (ONNX) + PostgreSQL/pgvector  
**Data Sovereignty:** 100% Air-Gapped Local Inference (0 Bytes Cloud Egress)  

---

## 1. Executive Benchmark Summary

| Evaluation Dimension | Baseline Dense Retrieval | Hybrid RAG (RRF $k=60$) | Hybrid + ONNX Cross-Encoder | Net Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Hit Rate @ 5** | 100.0% | 100.0% | **100.0%** | Reliable candidate recall |
| **Precision @ 5** | 70.0% | 70.0% | **89.9%** | **+19.9%** relevant chunk density |
| **Mean Reciprocal Rank (MRR@5)** | 0.875 | 0.875 | **1.000** | **+14.3%** top-rank guarantee |
| **Ranking Quality (nDCG@5)** | 0.903 | 0.903 | **0.980** | **+8.5%** near-ideal ordering |
| **Answer Grounding & Correctness** | 75.0% | 83.3% | **93.3%** | Verifiable source attribution |
| **Hallucination Rate** | 18.2% | 8.3% | **0.0%** | Zero hallucinations detected |
| **Refusal Accuracy on Out-of-Context** | 33.3% | 66.7% | **100.0%** | Honest refusals across probes |
| **Pre-Generation Retrieval Latency** | 940.9 ms | 386.0 ms | **1,621.5 ms** | Includes CPU Cross-Encoder |
| **Median TTFT (P50)** | 6,850 ms | 7,120 ms | **8,055.7 ms** | First token dispatch on RTX 3050 |
| **Average Generation Speed** | 22.4 TPS | 24.1 TPS | **25.7 TPS** | Peaks at 49.0 TPS (Warm Cache) |
| **GPU VRAM Ceiling Margin** | 2,807 MB | 2,807 MB | **2,898 MB / 4,096 MB** | **29.3%** safety headroom preserved |

---

## 2. Latency & Resource Breakdown

| Pipeline Stage | Implementation Detail | P50 Latency | P95 Latency | Mean Latency |
| :--- | :--- | :---: | :---: | :---: |
| **1. Embedding Generation** | `nomic-embed-text` (768-dim) via Ollama | 250 ms | 480 ms | 310 ms |
| **2. Dense Vector Search** | pgvector HNSW cosine index ($m=16, ef=64$) | 15 ms | 28 ms | 19 ms |
| **3. Sparse Lexical Search** | PostgreSQL full-text search (`tsvector`, GIN) | 16 ms | 35 ms | 23 ms |
| **4. Reciprocal Rank Fusion** | Rank fusion algorithm ($k=60$) | 2 ms | 4 ms | 3 ms |
| **5. Cross-Encoder Reranking** | `bge-reranker-base` on ONNX Runtime (CPU) | 858 ms | 1,599 ms | 1,180 ms |
| **6. State Checkpoint Load** | PostgreSQL JSONB state retrieval | 1 ms | 3 ms | 2 ms |
| **7. Time-to-First-Token (TTFT)** | Ollama GPU prompt ingestion & prefill | **8,055.7 ms** | **31,851.0 ms** | **8,746.4 ms** |
| **8. Autoregressive Generation** | Local 3B token streaming | **11,870 ms** | **53,742 ms** | **16,925 ms** |
| **Total End-to-End Latency** | Full roundtrip query response | **18.56 s** | **59.40 s** | **25.67 s** |

---

## 3. Resume & Portfolio Bullet Points

These bullet points are quantified, ATS-optimized, and ready for use in senior machine learning, AI engineer, and full-stack resumes:

### Option A — RAG & Retrieval Systems Focus
> - **Architected an air-gapped local sovereign RAG engine** combining PostgreSQL/pgvector HNSW indexes, full-text lexical search, and ONNX Runtime cross-encoder reranking (`BAAI/bge-reranker-base`), elevating **nDCG@5 from 0.903 to 0.980 (+8.5%)** and **Precision@5 from 70.0% to 89.9% (+19.9%)**.
> - **Engineered an adversarial refusal and grounding pipeline** on a local NVIDIA RTX 3050 GPU (4GB VRAM), achieving **93.3% answer correctness**, **0.0% hallucination rate**, and **100% honest refusal** across diverse multi-document benchmark queries.
> - **Benchmarked and profiled on-device inference telemetry** with Ollama (Llama 3.2 3B), achieving **25.7 tokens/second generation throughput** (bursting to 49.0 TPS) while preserving a **29.3% VRAM safety headroom** (2,898 MB peak allocation of 4,096 MB).

### Option B — AI Infrastructure & Full-Stack Focus
> - **Built TARK Studio**, a local-first enterprise AI workspace with dual-engine auto-routing, streaming SSE responses, and LangGraph-orchestrated stateful workflows backed by PostgreSQL checkpointers.
> - **Optimized local hybrid retrieval latency to 386ms** using Reciprocal Rank Fusion ($k=60$) across dense pgvector embeddings and sparse tsvector indexes, cutting baseline query latency by 59%.
> - **Authored automated benchmark harness and mathematical IR audit** (Hit Rate@K, MRR@K, nDCG@K, P50/P95 latencies) covering 15 diverse technical queries and 5 distinct document corpora with 0 bytes cloud egress.

---

## 4. Key Engineering Takeaways

1. **Cross-Encoders Are Essential for High-Value Retrieval:**  
   Dense vector embeddings alone achieved an nDCG@5 of 0.903. Adding an ONNX Runtime cross-encoder promoted the primary ground-truth chunk to rank #1 in 100% of factual tests (MRR 1.000).
2. **Strict Context Labeling Eliminates False Refusals:**  
   Injecting provenance metadata (`[Document: <name> | Page X]`) into retrieved prompt excerpts eliminated model confusion regarding source identity and resolved cross-document retrieval failures.
3. **4GB VRAM Is Sufficient for Sovereign Workloads When Memory-Budgeted:**  
   By offloading the cross-encoder to CPU and utilizing 4-bit quantized generative weights, the system runs reliably within 2.9GB of VRAM without crashing or swapping to disk.
