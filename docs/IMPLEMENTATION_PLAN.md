# TARK Studio — Implementation Roadmap & Execution Plan

**Version:** 1.0.0  
**Status:** Architecture & Planning Phase (Awaiting Review)  
**Governing Standard:** AGENTS.md & EMBER SAGE Design Identity

---

## 1. Proposed Project Directory Structure

The structure is minimal, strictly decoupled, and avoids premature micro-abstractions:

```
tark-studio/
├── AGENTS.md                     # Project-wide engineering specification
├── docs/                         # Architecture, PRD, DB, API, Plan specifications
│   ├── PRODUCT_REQUIREMENTS.md
│   ├── ARCHITECTURE.md
│   ├── DATABASE_DESIGN.md
│   ├── API_CONTRACTS.md
│   └── IMPLEMENTATION_PLAN.md
├── backend/                      # Python 3.12+ FastAPI backend service
│   ├── alembic/                  # Database migration scripts
│   │   ├── versions/
│   │   └── env.py
│   ├── alembic.ini
│   ├── pyproject.toml            # Dependencies and build configuration
│   ├── app/
│   │   ├── main.py               # Application entrypoint & lifespan
│   │   ├── config.py             # Pydantic BaseSettings environment config
│   │   ├── api/                  # Route endpoints
│   │   │   └── v1/
│   │   │       ├── router.py
│   │   │       ├── conversations.py
│   │   │       ├── messages.py
│   │   │       ├── models.py
│   │   │       ├── documents.py
│   │   │       └── memories.py
│   │   ├── core/                 # Shared errors, security, SSE helpers
│   │   │   ├── errors.py
│   │   │   ├── security.py
│   │   │   └── events.py
│   │   ├── db/                   # Database engine & base
│   │   │   ├── session.py
│   │   │   └── base.py
│   │   ├── models/               # SQLAlchemy ORM declarations
│   │   │   ├── conversation.py
│   │   │   ├── message.py
│   │   │   ├── document.py
│   │   │   ├── chunk.py
│   │   │   └── memory.py
│   │   ├── schemas/              # Pydantic v2 schemas / DTOs
│   │   │   ├── conversation.py
│   │   │   ├── message.py
│   │   │   ├── model.py
│   │   │   ├── document.py
│   │   │   └── memory.py
│   │   ├── services/             # Core business logic
│   │   │   ├── conversation_service.py
│   │   │   ├── document_service.py
│   │   │   ├── memory_service.py
│   │   │   └── ingestion_service.py
│   │   └── ai/                   # AI orchestration and model adapters
│   │       ├── providers/
│   │       │   ├── base.py
│   │       │   ├── ollama.py
│   │       │   ├── groq.py
│   │       │   └── gemini.py
│   │       ├── embeddings/
│   │       │   ├── base.py
│   │       │   └── provider.py
│   │       └── workflows/
│   │           ├── state.py
│   │           └── chat_workflow.py
│   └── tests/                    # Pytest test suite
│       ├── conftest.py
│       ├── test_conversations.py
│       ├── test_providers.py
│       ├── test_rag_pipeline.py
│       └── test_memory.py
├── frontend/                     # React / Vite / TypeScript client
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── index.css             # EMBER SAGE theme tokens & global styles
│   │   ├── main.tsx              # React mounting root
│   │   ├── App.tsx               # Top-level workspace layout
│   │   ├── types/                # TypeScript interfaces matching API DTOs
│   │   │   ├── chat.ts
│   │   │   ├── document.ts
│   │   │   └── memory.ts
│   │   ├── services/             # API client & SSE consumer
│   │   │   ├── api.ts
│   │   │   └── stream.ts
│   │   ├── hooks/                # Custom React hooks
│   │   │   ├── useChatStream.ts
│   │   │   ├── useConversations.ts
│   │   │   ├── useDocuments.ts
│   │   │   └── useMemories.ts
│   │   └── components/           # Neo-brutalist UI components
│   │       ├── layout/
│   │       │   ├── Sidebar.tsx
│   │       │   └── Header.tsx
│   │       ├── chat/
│   │       │   ├── MessageList.tsx
│   │       │   ├── MessageBubble.tsx
│   │       │   ├── CodeBlock.tsx
│   │       │   ├── MarkdownRenderer.tsx
│   │       │   └── PromptInput.tsx
│   │       ├── panels/
│   │       │   ├── DocumentDrawer.tsx
│   │       │   └── MemoryDrawer.tsx
│   │       └── common/
│   │           ├── Button.tsx
│   │           ├── Badge.tsx
│   │           └── Modal.tsx
│   └── tests/                    # Vitest component & hook tests
│       ├── useChatStream.test.ts
│       └── MessageBubble.test.tsx
└── docker-compose.yml            # Local PostgreSQL 16 + pgvector container
```

---

## 2. Implementation Milestones

### Milestone 0: Architecture, Planning & Approval (Current Milestone)
- **Scope:** Define product scope, architecture, database schemas, API contracts, folder layout, and review open technical questions.
- **Deliverables:** Completed `docs/` documentation suite.
- **Acceptance Criteria:**
  - All 5 documents authored and cross-referenced.
  - Zero code written until explicit user review and go-ahead.

---

### Milestone 1: Workspace Initialisation & Database Foundation
- **Scope:** Initialize project skeleton, create Docker Compose configuration with `pgvector/pgvector:pg16` (including persistent volume and healthcheck), configure environment-driven asynchronous database connectivity, and establish Alembic migrations enabling `vector` and `uuid-ossp` extensions.
- **Key Deliverables:**
  - `docker-compose.yml` defining PostgreSQL 16 with `pgvector/pgvector:pg16`, persistent named volume (`tark_postgres_data`), and `pg_isready` healthcheck.
  - Backend project scaffolding with `pyproject.toml` (FastAPI, SQLAlchemy, asyncpg, alembic, pydantic-settings).
  - Environment configuration (`.env.example` and `config.py`) supporting custom PostgreSQL connection strings without hardcoded credentials.
  - Alembic async environment and initial migration script (`0001_initial_extensions.py`) enabling `uuid-ossp` and `vector`.
  - Frontend skeleton (Vite, React 18, TypeScript, Tailwind CSS with EMBER SAGE tokens).
  - Pytest fixture and connectivity verification test verifying async PostgreSQL connection and extension status.
- **Acceptance Criteria:**
  - `docker compose up -d` brings up PostgreSQL with passing healthcheck.
  - `alembic upgrade head` executes cleanly on the database.
  - Test suite verifies database connectivity and validates `vector` extension presence.
  - Zero application chat, RAG, or memory code implemented yet.

---

### Milestone 2: Provider Abstraction & Core Chat Backend
- **Scope:** Implement the unified provider gateway (`BaseLLMProvider`) supporting Ollama, Groq, and Gemini; implement LangGraph basic chat state workflow; implement REST and SSE endpoints for conversation and message management.
- **Key Deliverables:**
  - Provider adapters: `OllamaProvider`, `GroqProvider`, `GeminiProvider`.
  - Privacy guard preventing cloud execution for local-only threads.
  - FastAPI endpoints: `GET/POST/PATCH/DELETE /api/v1/conversations`, `POST /api/v1/conversations/{id}/messages` (SSE streaming).
  - Cancellation handling via client disconnection detection.
- **Acceptance Criteria:**
  - Unit tests verify all three providers adhere to `BaseLLMProvider` interface.
  - Mocked and live Ollama tests confirm token streaming via SSE.
  - Aborting an active stream cleanly halts downstream generation and updates message finish reason to `abort`.

---

### Milestone 3: Frontend Shell & Real-Time Chat Interface (EMBER SAGE)
- **Scope:** Initialize Vite + React + TypeScript + Tailwind CSS client; configure EMBER SAGE theme tokens; implement sidebar, conversation list, model selector, message list, and auto-expanding prompt dock with SSE streaming.
- **Key Deliverables:**
  - Tailwind CSS configuration with EMBER SAGE palette (`#10130F`, `#171C17`, `#20271F`, `#394336`, `#A5B49A`, `#E59A64`).
  - Typography setup for `Instrument Serif`, `Instrument Sans`, and `IBM Plex Mono`.
  - Custom `useChatStream` hook handling SSE chunk accumulation, abort signals, and error states.
  - Accessible, neo-brutalist message bubbles with syntax-highlighted code blocks and copy buttons.
- **Acceptance Criteria:**
  - User can create a conversation, select a model, submit a prompt, and observe token streaming without UI stutter.
  - Color contrast meets WCAG 2.2 AA on all rendered components.
  - Vitest test suite passes for streaming state updates.

---

### Milestone 4: Persistent Memory Subsystem
- **Scope:** Implement background memory extraction in LangGraph; build vector retrieval for active memories; add frontend Memory Management Drawer.
- **Key Deliverables:**
  - LangGraph `ExtractMemoryNode` evaluating completed turns.
  - Vector similarity retrieval querying `memories` table during prompt composition.
  - REST endpoints: `GET/POST/PATCH/DELETE /api/v1/memories`.
  - Frontend Memory Drawer to inspect, toggle active status, edit, or purge memories.
- **Acceptance Criteria:**
  - Stated preferences (e.g., *"Prefer concise answers"*) are automatically extracted and saved to PostgreSQL.
  - Subsequent queries retrieve and inject the preference, and the SSE stream emits a `memory_applied` event.
  - User can view and delete memories in the frontend.

---

### Milestone 5: Document Ingestion & Grounded RAG Pipeline
- **Scope:** Implement file upload, document parsing (PDF, Markdown, TXT), recursive token chunking, vector embedding, and HNSW semantic retrieval with bounded prompt injection and citation display.
- **Key Deliverables:**
  - Document ingestion service (`IngestionService`) handling validation, chunking, and vector embedding.
  - LangGraph `RetrieveDocsNode` injecting bounded context.
  - REST endpoints: `POST /api/v1/documents/upload`, `GET/DELETE /api/v1/documents`, attachment junction endpoints.
  - Frontend Document Drawer & drag-and-drop file upload tray.
  - Interactive citation badges in assistant messages linking directly to source text snippets.
- **Acceptance Criteria:**
  - PDF and Markdown files upload, chunk, and embed within 5 seconds for typical 10-page documents.
  - RAG queries return grounded answers with clickable citation pills.
  - Prompt injection boundary tests verify that malicious instructions inside ingested documents are ignored by the model.

---

### Milestone 6: Hardening, Polish, Testing & Accessibility
- **Scope:** Complete end-to-end integration tests; verify keyboard shortcuts (`Cmd+K`, `Escape`); audit reduced motion; perform benchmark tests on TTFT and streaming throughput.
- **Key Deliverables:**
  - Comprehensive Pytest backend suite (>85% coverage).
  - Vitest frontend component suite.
  - Production build verification (`npm run build`, `python -m pytest`).
- **Acceptance Criteria:**
  - Full automated test suite passes cleanly.
  - Zero unhandled exceptions or console errors.
  - Ready for milestone review and production sign-off.

---

## 3. Dependency Inventory & Justifications

### Backend Dependencies (`pyproject.toml`)
| Dependency | Version Spec | Technical Justification |
| :--- | :--- | :--- |
| `fastapi` | `^0.115.0` | High-performance async ASGI web framework for REST and SSE. |
| `uvicorn[standard]` | `^0.32.0` | Production-ready ASGI server supporting HTTP/1.1 and WebSockets. |
| `pydantic` | `^2.9.0` | Strict type validation, DTO serialization, and settings management. |
| `pydantic-settings` | `^2.6.0` | Twelve-factor environment variable loading from `.env`. |
| `sqlalchemy[asyncio]`| `^2.0.36` | Modern asynchronous ORM and Core SQL builder. |
| `asyncpg` | `^0.30.0` | High-performance asynchronous PostgreSQL driver for Python. |
| `pgvector` | `^0.3.6` | Official Python helper library for PostgreSQL vector operations. |
| `alembic` | `^1.14.0` | Deterministic, transactional database migration tool. |
| `langgraph` | `^0.2.50` | Stateful AI orchestration graph for multi-step retrieval & memory. |
| `httpx` | `^0.27.2` | Async HTTP client for communicating with local Ollama daemon. |
| `groq` | `^0.13.0` | Official high-speed client for Groq cloud inference. |
| `google-genai` | `^0.1.1` | Official Google GenAI SDK for Gemini 1.5/2.0 models. |
| `pypdf` | `^5.1.0` | Fast, dependency-free PDF text extraction. |
| `tiktoken` | `^0.8.0` | Fast token counting for exact chunk sizing. |
| `pytest` / `pytest-asyncio` | `^8.3.0` | Test runner and async testing harness. |

### Frontend Dependencies (`package.json`)
| Dependency | Technical Justification |
| :--- | :--- |
| `react`, `react-dom` (`^18.3.1`) | UI library for declarative view layer. |
| `typescript` (`^5.6.0`) | Compile-time type safety aligned with backend schemas. |
| `vite` (`^5.4.0`) | Instant HMR development server and optimized rollup bundler. |
| `tailwindcss` (`^3.4.15`) | Utility-first CSS configured with custom EMBER SAGE design tokens. |
| `lucide-react` (`^0.460.0`) | Crisp, lightweight geometric SVG icon library. |
| `react-markdown` (`^9.0.1`) | Secure, component-based Markdown parser. |
| `remark-gfm` (`^4.0.0`) | GitHub Flavored Markdown support (tables, autolinks, task lists). |
| `prismjs` (`^1.29.0`) | Lightweight client-side code syntax highlighter. |
| `vitest` (`^2.1.5`) | Vite-native unit testing runner. |

---

## 4. Technical Risks & Mitigation Strategies

| Risk | Impact | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Local Ollama Daemon Offline** | Medium | High | Backend performs non-blocking health checks. If Ollama is offline, the UI flags local models as "offline" with troubleshooting guidance rather than crashing. |
| **Embedding Dimension Mismatch** | High | Low | Lock standard vector dimensions to 768. All embeddings (Ollama `nomic-embed-text` or Google text-embedding) map cleanly to `VECTOR(768)`. |
| **Memory Extraction Hallucinations** | Medium | Medium | LangGraph memory node employs strict JSON schema validation and a confidence score threshold ($\ge 0.85$). Users can inspect and delete any extracted memory in the UI. |
| **Prompt Injection via Ingested Docs** | Critical | Medium | Bounded context isolation using `<untrusted_context>` tags; system prompt explicitly commands model to ignore prompt-override attempts in retrieved text. |
| **SSE Connection Drops during Generation** | Medium | Medium | Frontend tracks last received message chunk index. Backend detects `is_disconnected()` to immediately cancel orphaned inference tasks. |

---

## 5. Adopted Technical Decisions

The following architectural decisions have been confirmed and integrated:

1. **Embeddings:**
   - Ollama `nomic-embed-text` adopted as the initial candidate with 768 vector dimensions.
   - The model is not installed or loaded during Milestone 1.
   - Retrieval quality claims will be verified through empirical benchmarks during the RAG milestone, not assumed.

2. **Database:**
   - Docker Compose uses official `pgvector/pgvector:pg16` with named persistent volume (`tark_postgres_data`) and container healthcheck.
   - Vector extension is enabled via Alembic migrations.
   - Custom PostgreSQL connections are supported via environment configuration (`DATABASE_URL`). Credentials are never hardcoded.

3. **Memory:**
   - Memory extraction implementation is deferred to the dedicated Memory milestone.
   - No premature queue infrastructure (no Redis or Celery).
   - Extraction will be designed as an asynchronous, idempotent, and retryable background operation.

4. **Document RAG Formats:**
   - Initial formats strictly limited to PDF, Markdown (`.md`), Plain Text (`.txt`), Python (`.py`), and JSON (`.json`).
   - Word (`.docx`) and Excel (`.xlsx`) are kept out of initial scope.

5. **Streaming Protocol:**
   - Client uses fetch-based streaming with `ReadableStream` for POST message dispatches.
   - AbortController cancellation propagates to backend model generation tasks immediately.
   - Reconnect behavior and turn resume are explicitly managed via monotonic sequence IDs.

6. **Architecture & Scope Isolation:**
   - Normal CRUD and database operations remain in standard FastAPI/SQLAlchemy services outside LangGraph.
   - LangGraph is reserved strictly for stateful AI generation and retrieval flows when required.
