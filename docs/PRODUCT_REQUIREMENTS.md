# TARK Studio — Product Requirements Document (PRD)

**Version:** 1.0.0  
**Status:** Architecture & Planning Phase (Awaiting Review)  
**Author:** TARK Engineering & Product Team  
**Design Identity:** EMBER SAGE (Dark Neo-Brutalism / Premium Industrial)

---

## 1. Executive Summary & Product Vision

**TARK Studio** is a premium, local-first artificial intelligence workspace engineered for engineers, researchers, and technical power users. It unifies state-of-the-art cloud inference (Groq for ultra-low latency, Google Gemini for deep reasoning and broad context) with local inference (Ollama) under a unified, high-performance interface.

Unlike generic consumer chat interfaces that prioritize decorative animations and opaque cloud storage, TARK Studio is built on the following foundational tenets:
- **Local-First & Data Sovereignty:** Core conversation data, document vectors, and personal memories reside on the user's infrastructure in PostgreSQL (`pgvector`). Local inference with Ollama ensures zero data exfiltration.
- **Deterministic Multi-Provider Orchestration:** Explicit, intentional model selection across local (Ollama) and cloud (Groq, Gemini) providers. Never silently degrade or switch execution environments.
- **Persistent Context & Stateful Workflows:** State machines powered by LangGraph for reliable context management, semantic memory recall, and document-grounded Retrieval-Augmented Generation (RAG).
- **Distinctive Industrial Aesthetics (EMBER SAGE):** High-contrast, utilitarian design identity adhering to dark neo-brutalism, editorial typography, and strict WCAG 2.2 AA accessibility standards.

---

## 2. Design Identity & Interface System (EMBER SAGE)

The application visual system is governed strictly by the **EMBER SAGE** design specification:

### 2.1 Color Tokens
| Token Name | Hex Code | Visual Role & Usage |
| :--- | :--- | :--- |
| **Main Background** | `#10130F` | Primary application surface, editor body, message canvas (70% dominance) |
| **Sidebar Surface** | `#171C17` | Navigation panel, document tray, thread history background |
| **Elevated Surface** | `#20271F` | Cards, active thread items, popovers, model selectors, code blocks |
| **Border & Dividers** | `#394336` | 1px structural neo-brutalist borders, panel separation, divider rules |
| **Primary Text** | `#F1EEE6` | Headings, user prompt text, high-emphasis content (WCAG AAA contrast) |
| **Secondary Text** | `#B7BDAE` | Assistant message body, UI labels, navigation text |
| **Muted Text** | `#929B89` | Timestamps, token counters, file sizes, disabled states |
| **Sage Accent** | `#A5B49A` | Secondary emphasis, success states, verified citations, memory pills (20% share) |
| **Ember Accent** | `#E59A64` | Primary actions, send button, active model indicator, focus rings (10% accent) |
| **Ember Hover** | `#F0AB78` | Interactive hover state for Ember controls and primary CTA buttons |

### 2.2 Typography
- **Editorial Headlines:** `Instrument Serif` (Headings, modal titles, welcome banners, thread titles).
- **Application Interface:** `Instrument Sans` (Navigation, message body, button labels, settings).
- **Technical Metadata & Code:** `IBM Plex Mono` (Code blocks, token consumption metrics, latency indicators, model identifiers, vector scores).

### 2.3 Visual Rules
1. **70/20/10 Rule:** 70% deep organic darks (`#10130F`, `#171C17`), 20% structural sage (`#20271F`, `#394336`, `#A5B49A`), 10% purposeful copper/ember (`#E59A64`).
2. **Neo-Brutalist Geometry:** Crisp 1px solid borders (`#394336`), sharp corners (2px to 4px radius max), high contrast, deliberate absence of soft drop-shadows or blurry glassmorphism.
3. **No AI Clichés:** Strictly forbidden are purple-to-blue neon gradients, floating orb animations, and decorative confetti.
4. **Accessibility:** Minimum contrast ratio of 4.5:1 for body copy and 3:1 for large display text against dark backgrounds. Full keyboard navigation and reduced-motion support.

---

## 3. Target Persona & Core User Workflows

### 3.1 Primary Persona: The Privacy-Conscious Technical Practitioner
- Software engineers, security analysts, and researchers working with proprietary code, internal architecture notes, or sensitive documentation.
- Demands local inference for sensitive inputs, yet wants immediate access to high-speed cloud inference (Groq Llama 3) or massive context windows (Gemini 1.5/2.0) when appropriate.
- Relies on persistent memory to avoid repeating project constraints, coding conventions, and architectural preferences in every session.

### 3.2 Core User Journeys

#### Journey 1: Local-to-Cloud Chat Switching & Real-Time Streaming
1. User opens TARK Studio. The system defaults to the last-used model (or local Ollama model if configured).
2. User enters a query into the prompt input. An explicit pill tag indicates whether the target provider is **Local (Ollama)** or **Cloud (Groq/Gemini)**.
3. User dispatches the prompt (`Enter` or `Ctrl+Enter`).
4. Backend verifies provider health, initializes the LangGraph execution flow, and streams tokens over Server-Sent Events (SSE).
5. Frontend renders tokens immediately with zero UI freeze. Monospace code snippets are highlighted incrementally.
6. Upon stream completion, metadata (Time to First Token [TTFT], tokens per second, model ID) renders in a subtle monospace bar.

#### Journey 2: Document Ingestion & Grounded RAG Query
1. User drops a technical specification PDF/Markdown file onto the RAG panel or active thread.
2. The file is uploaded to the backend, validated, parsed, chunked, and embedded into PostgreSQL via `pgvector`.
3. Ingestion progress is displayed with chunk counts and index confirmation.
4. User asks: *"What are the token limits defined in Section 3?"*
5. The LangGraph pipeline retrieves the top-k semantically relevant chunks, appends them as bounded context to the system prompt, and streams the response.
6. The assistant output includes clickable citation pills referencing the exact document title and section/chunk. Clicking a citation highlights the exact excerpt in a dedicated inspector drawer.

#### Journey 3: Continuous Memory Formation & Inspection
1. During a multi-turn conversation, user mentions: *"I am building our backend with Python 3.12 and strictly enforce async SQLAlchemy 2.0 without ORM lazy-loading."*
2. LangGraph's post-turn memory extraction node identifies this as a persistent technical preference.
3. A memory record is created in PostgreSQL with an embedding and category (`preference`).
4. A subtle indicator in the UI notes: *"New memory retained: Prefers async SQLAlchemy 2.0"*.
5. In subsequent threads, even days later, the memory is semantically retrieved and injected, ensuring all generated code follows async SQLAlchemy conventions without re-prompting.
6. User can open the Memory Drawer at any time to inspect, edit, or purge retained memories.

---

## 4. Functional Requirements

### 4.1 Chat & Conversation Management (FR-CHAT)
- **FR-CHAT-01:** System shall support creating, renaming, searching, archiving, and deleting conversation threads.
- **FR-CHAT-02:** System shall persist all user and assistant messages, including system prompts and model metadata, in PostgreSQL.
- **FR-CHAT-03:** System shall support message regeneration from any assistant turn, maintaining branching history or replacing the latest response.
- **FR-CHAT-04:** System shall support copying message contents, formatted markdown, and isolated code blocks with one click.
- **FR-CHAT-05:** System shall support stopping active token generation via client abort (`AbortController`), triggering immediate backend task cancellation.

### 4.2 Multi-Provider Model Orchestration (FR-MODEL)
- **FR-MODEL-01:** System shall support three distinct providers via an extensible abstraction layer:
  - **Ollama:** Local inference (e.g., `llama3.2`, `qwen2.5-coder`, `mistral`).
  - **Groq:** Cloud ultra-low-latency inference (e.g., `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`).
  - **Gemini:** Cloud deep reasoning and massive context (e.g., `gemini-1.5-pro`, `gemini-2.0-flash`).
- **FR-MODEL-02:** System shall dynamically query available models from local Ollama instances and validate API keys for cloud providers.
- **FR-MODEL-03:** System shall enforce a strict **Privacy Boundary**: Threads flagged as `local_only` shall strictly forbid dispatching data to Groq or Gemini endpoints, raising a deterministic client error if a cloud model is selected.
- **FR-MODEL-04:** System shall display real-time latency indicators (TTFT, total generation time, tokens/sec).

### 4.3 Streaming Architecture (FR-STREAM)
- **FR-STREAM-01:** Chat message dispatch and token streaming must use client-initiated `fetch` with `ReadableStream` consuming Server-Sent Events (`text/event-stream`).
- **FR-STREAM-02:** Event stream must emit distinct event types:
  - `token`: Partial text deltas.
  - `citation`: Grounded document sources and similarity scores.
  - `memory_applied`: Identified memories active in current context.
  - `meta`: Token usage metrics, execution latency, finish reason.
  - `error`: Structured error codes and human-readable explanations.
  - `done`: Explicit signal indicating end of stream.
- **FR-STREAM-03:** Frontend abort controller cancellation must propagate directly to backend request context, halting active model generation immediately. Reconnect and replay logic must be explicitly managed by the fetch stream consumer using monotonic sequence IDs rather than relying on browser-native EventSource reconnects.
- **FR-STREAM-04:** Backend must handle client disconnects safely, discard orphaned partial responses, and prevent duplicate in-flight requests for the same conversation turn.

### 4.4 Persistent Memory Subsystem (FR-MEM)
- **FR-MEM-01:** System shall maintain an autonomous semantic memory store backed by `pgvector`.
- **FR-MEM-02:** Memory extraction implementation is deferred to the dedicated Memory milestone. When implemented, extraction shall run as an asynchronous, idempotent, and retryable background operation without introducing external queue infrastructure (e.g., Redis/Celery) prematurely.
- **FR-MEM-03:** Extracted memories shall be categorized (`preference`, `fact`, `constraint`, `instruction`).
- **FR-MEM-04:** For each user prompt, system shall perform vector similarity search over memories and inject relevant items into the orchestration context.
- **FR-MEM-05:** System shall provide a dedicated management interface to view, update confidence scores, toggle active state, or permanently delete memories.

### 4.5 Document Management & RAG (FR-RAG)
- **FR-RAG-01:** System shall initially support uploading PDF, Markdown (`.md`), Plain Text (`.txt`), Python (`.py`), and JSON (`.json`) files. Proprietary Office formats (`.docx`, `.xlsx`) are strictly out of initial scope.
- **FR-RAG-02:** Files must be validated for MIME type, size limit (default 25MB per file), and scanned for corrupt structures.
- **FR-RAG-03:** Ingestion pipeline must chunk documents using recursive token-aware chunking with customizable chunk size (default 512 tokens) and overlap (default 64 tokens).
- **FR-RAG-04:** Chunks must be vectorized using the designated embedding model candidate (Ollama `nomic-embed-text`, 768 dimensions) and stored in PostgreSQL using `pgvector` with HNSW indexing. (Note: Embedding model will not be loaded or evaluated until the RAG milestone, and retrieval performance claims will be benchmark-verified rather than assumed).
- **FR-RAG-05:** System shall support scoping RAG to specific threads (associating specific documents with specific conversations) or querying a global document library.
- **FR-RAG-06:** Assistant responses must explicitly tag retrieved sources with verifiable citations.

---

## 5. Non-Functional Requirements

### 5.1 Performance & Latency
- **NFR-PERF-01 (TTFT):** Time to first token for cloud providers (Groq) must be under 600ms on broadband connections.
- **NFR-PERF-02 (Local Inference):** Ollama streaming pipeline overhead (FastAPI proxy + SSE dispatch) must introduce less than 15ms latency above native Ollama generation.
- **NFR-PERF-03 (RAG Retrieval):** Vector search over 100,000 document chunks using pgvector HNSW index must complete in under 50ms.
- **NFR-PERF-04 (Frontend Rendering):** Chat view must maintain 60 FPS while streaming tokens; Markdown parsing and syntax highlighting must use batched virtual updates.

### 5.2 Security & Data Privacy
- **NFR-SEC-01 (Data Isolation):** Local-mode conversations, document uploads, and vector embeddings must never leave the local environment unless cloud providers are explicitly targeted.
- **NFR-SEC-02 (Prompt Injection Defense):** Retrieved document content must be enclosed in strict XML/markdown boundaries (`<context_document id="...">...</context_document>`) with explicit system instructions prohibiting execution of enclosed instructions.
- **NFR-SEC-03 (Credential Safety):** API keys (Groq, Gemini) must be stored in server environment variables or encrypted configuration files—never committed to source control and never exposed in client API responses.
- **NFR-SEC-04 (Upload Sanitization):** Uploaded files must be processed in-memory or in isolated temporary directories with strict filename sanitization and path traversal prevention.

### 5.3 Reliability & Availability
- **NFR-REL-01:** Server must handle database connection timeouts and provider failures with graceful error events rather than unhandled 500 crashes.
- **NFR-REL-02:** Database schema changes must be fully managed through automated Alembic migrations with forward and backward compatibility.

### 5.4 Usability & Accessibility
- **NFR-ACC-01:** Comply with **WCAG 2.2 AA** contrast and navigation standards.
- **NFR-ACC-02:** Full keyboard navigation support (e.g. `Cmd/Ctrl + K` for search, `Cmd/Ctrl + /` for thread drawer, `Escape` to close drawers).
- **NFR-ACC-03:** Support `prefers-reduced-motion` across all transitions and animations.

---

## 6. Explicitly Excluded Features (Initial Milestone)

To ensure razor-sharp focus on foundational quality and stability, the following capabilities are **strictly out of scope** for this initial milestone:
1. **Live Web Search & Browsing:** No DuckDuckGo, Google, or Bing search tools.
2. **Model Context Protocol (MCP) Tools:** No external MCP servers, filesystem agents, or CLI tool integrations.
3. **Autonomous Agent Loops:** No open-ended multi-agent task planning, self-reflection loops, or autonomous subagents.
4. **Multi-Tenant User Management & Auth:** No OAuth2, Cognito, or multi-user tenant billing in v1 (single-user / desktop-style local-first architecture).
5. **In-Browser Code Execution Sandboxes:** No Pyodide or Docker-based live code runners. Code is formatted and highlighted only.
