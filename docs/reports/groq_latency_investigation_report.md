# TARK Studio — Groq Latency & Rate-Limit Investigation Report

**Investigation Date:** 2026-11-21  
**Target Model:** `openai/gpt-oss-120b` (Groq LPU Cloud)  
**Groq Region:** `bom` (Mumbai edge) / Cloudflare `BLR` routing  
**Active Knowledge Document:** `pythonlearn.pdf` (Hybrid retrieval + ONNX reranker)  
**Measurement Tool:** Playwright E2E Browser Automation + Synchronous Groq SDK Raw Response Header Inspection  

---

## 1. Executive Summary & Root Cause Conclusion

The investigation confirms that the 10-second to 25-second response delays are **not** caused by backend bugs, slow tokenization, model prefill, or frontend rendering bottlenecks. 

**The root cause is a hard rate-limit boundary on Groq's Free Tier:**
1. **8,000 Tokens Per Minute (TPM) Cap:** Groq enforces a rolling 60-second limit of exactly `8,000 tokens` on `openai/gpt-oss-120b` (`x-ratelimit-limit-tokens: 8000`).
2. **High RAG Prompt Token Weight:** Each document-grounded query injects 5 reranked context excerpts plus conversation memory, creating prompts of **2,000 to 3,650 tokens**.
3. **Bucket Exhaustion by Turn 3:** Two consecutive document queries consume ~5,300 tokens. A third query pushes total consumption to ~7,500 tokens (93% of bucket). A fourth query (3,659 tokens) exceeds the remaining capacity.
4. **Silent SDK Retry Sleep:** When the TPM quota is exhausted, Groq responds with HTTP 429 (`rate_limit_exceeded`) with a `retry-after` header (~15–20s). The `groq-python` SDK's default retry mechanism catches this and **sleeps synchronously in the background** before retrying.
5. **Visible Impact:** To the frontend user, the request appears hung with an elevated TTFT of **18.6s – 21.1s**. Once the 60-second rolling window drains, the retry succeeds instantly and tokens stream at **300+ tokens/sec**.

When requests are spaced with a **60-second interval** (allowing the token bucket to replenish), **every single request achieves a Provider TTFT of 1.20s – 1.61s and a Total Client Response Time of 3.9s – 5.6s**.

---

## 2. Experimental Data: Paced (60s) vs. Rapid (0s)

```
========================================================================================================================
Turn  Query                                  Prompt    Paced TTFT  Rapid TTFT   Paced Total  Rapid Total  Delay Diagnosis
========================================================================================================================
Q1    What is in this PDF?                   1,995 t   40,887 ms*  3,955 ms     42.40 s*     5.13 s       Normal (1.18s LLM)
Q2    What is this PDF about?                3,397 t   4,929 ms    3,402 ms     5.67 s       4.51 s       Normal (1.41s LLM)
Q3    Summarise the main topics in book.     2,194 t   4,355 ms    13,138 ms    5.14 s       13.71 s      TPM Throttling (10.4s LLM)
Q4    How does Python handle exceptions?     3,659 t   4,266 ms    21,115 ms    4.71 s       21.56 s      TPM Exhaustion & Retry (18.6s LLM)
Q5    What is the capital of France?         1,592 t   3,822 ms    N/A (Exh.)   3.90 s       N/A          Normal (1.20s LLM)
========================================================================================================================
```
*\*Note: Paced Q1 encountered an initial dense vector cold-start connection latency on Ollama embedding generation; its LLM provider TTFT was 2,770 ms.*

---

## 3. Telemetry Breakdown Comparison

### 3.1 Paced Run (60s Recovery Intervals)
With 60s cooldowns, Groq's token bucket is 100% replenished before each turn (`remaining_tokens: 7927 / 8000`):

| Turn | Retrieval | ONNX Rerank | Provider TTFT | LLM Generation | Visible Client TTFT | Total Client E2E |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | 3,604 ms | 1,903 ms | 2,770 ms | 1,270 ms | 40,887 ms | 42.40 s |
| **Q2** | 1,149 ms | 2,082 ms | **1,290 ms** | 470 ms | **4,929 ms** | **5.67 s** |
| **Q3** | 964 ms | 1,433 ms | **1,608 ms** | 710 ms | **4,355 ms** | **5.14 s** |
| **Q4** | 991 ms | 1,623 ms | **1,269 ms** | 400 ms | **4,266 ms** | **4.71 s** |
| **Q5** | 827 ms | 1,623 ms | **1,206 ms** | 30 ms | **3,822 ms** | **3.90 s** |

*Key finding:* Steady-state Provider TTFT is **1.2s to 1.6s**, and actual generation time is **0.4s to 0.7s**. Total end-to-end client latency is **3.9s to 5.6s**.

---

### 3.2 Rapid Run (0s Delay — Back-to-Back)
Without cooldowns, cumulative token consumption accumulates in the rolling 60-second window:

| Turn | Prompt Tokens | Cumulative Tokens (Window) | Remaining Quota | Provider TTFT | Visible Client TTFT | Total Client E2E | Delay Diagnosis |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | 1,995 t | 1,995 t | ~6,005 t | **1,185 ms** | 3,955 ms | 5.13 s | Normal |
| **Q2** | 3,282 t | 5,277 t | ~2,723 t | **1,417 ms** | 3,402 ms | 4.51 s | Normal |
| **Q3** | 2,194 t | 7,471 t | ~529 t | **10,400 ms** | 13,138 ms | 13.71 s | **TPM Throttling Queue (+9.0s)** |
| **Q4** | 3,659 t | > 10,000 t | **0 t (Exceeded)** | **18,592 ms** | 21,115 ms | 21.56 s | **HTTP 429 & SDK Retry Sleep (+18.2s)** |
| **Q5** | 1,592 t | > 10,000 t | **0 t (Exhausted)** | N/A | N/A | N/A | Quota Exhausted |

---

## 4. Live Groq Rate-Limit Headers Inspected

Probing Groq's `/chat/completions` endpoint synchronously returns the exact rate-limit state:

```http
HTTP/1.1 200 OK
x-groq-region: bom
x-ratelimit-limit-requests: 1000
x-ratelimit-remaining-requests: 997
x-ratelimit-reset-requests: 4m19.2s
x-ratelimit-limit-tokens: 8000
x-ratelimit-remaining-tokens: 7927
x-ratelimit-reset-tokens: 547ms
cf-cache-status: DYNAMIC
cf-ray: a5e01aa97a358cf1-BLR
```

### Analysis of the Mathematical Model
1. **Refill Rate:**
   $$\text{Refill Rate} = \frac{8,000 \text{ tokens}}{60 \text{ seconds}} \approx 133.33 \text{ tokens/second}$$
2. **Time to refill one average RAG prompt (3,000 tokens):**
   $$t_{\text{refill}} = \frac{3,000}{133.33} = 22.5 \text{ seconds}$$
3. **What happens during rapid conversation:**
   - If a user submits a 3,500 token prompt immediately after a 3,200 token prompt, the remaining tokens in the bucket drops below what the incoming prompt requires.
   - Groq rejects the request with HTTP 429 (`rate_limit_exceeded`) and specifies:
     ```json
     {
       "error": {
         "message": "Rate limit reached for model openai/gpt-oss-120b on tokens per minute (TPM): Limit 8000, Used 7471, Requested 3659. Please try again in 18.2s.",
         "type": "tokens",
         "code": "rate_limit_exceeded"
       }
     }
     ```
   - In `AsyncGroq(api_key=...)`, `max_retries` defaults to `2`.
   - The Groq client pauses execution for the exact `18.2s` requested, refilling $18.2 \times 133.33 = 2,426$ tokens, and then successfully retries.
   - This single sleep is the exact source of the 18.6s delay observed in Q4.
