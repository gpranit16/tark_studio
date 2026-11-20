
# TARK Studio — Project Instructions

## 1. Product Vision
TARK Studio is a premium, local-first AI workspace.
Build a reliable foundation for chat, conversation history,
persistent memory and document-based RAG.

Initial scope:
- AI chat and thread management
- Multi-provider model support
- Streaming responses
- Persistent conversation storage
- Memory
- Document RAG

Do not implement web search, MCP tools or autonomous agents
in the initial milestone.

## 2. Technology Stack

Frontend:
- React
- Vite
- TypeScript
- Tailwind CSS

Backend:
- Python
- FastAPI
- Pydantic
- SQLAlchemy
- Alembic

AI:
- LangGraph for stateful AI workflows
- Groq, Gemini and Ollama providers

Database:
- PostgreSQL
- pgvector

Communication:
- REST APIs
- Server-Sent Events (SSE) for streaming

Testing:
- Pytest
- Vitest

## 3. Design Identity — EMBER SAGE

Visual direction:
Dark Neo-Brutalism, Premium Industrial,
Editorial Typography and restrained 3D visuals.

Colour tokens:
- Main background: #10130F
- Sidebar: #171C17
- Elevated surface: #20271F
- Borders: #394336
- Primary text: #F1EEE6
- Secondary text: #B7BDAE
- Muted text: #929B89
- Sage accent: #A5B49A
- Ember accent: #E59A64
- Ember hover: #F0AB78

Typography:
- Instrument Serif: editorial headlines
- Instrument Sans: application interface
- IBM Plex Mono: technical metadata and code

Visual rules:
- Preserve the 70/20/10 visual colour hierarchy.
- Use copper for primary actions and important highlights.
- Use sage for secondary emphasis and positive states.
- Avoid generic purple AI gradients.
- Avoid excessive glassmorphism and decorative effects.
- Use 3D elements selectively.
- Prioritise readability, accessibility and performance.
- Support responsive layouts and reduced motion.
- Target WCAG 2.2 AA contrast requirements.

## 4. Engineering Principles

- Prefer simple, readable and maintainable code.
- Use type hints and Pydantic validation.
- Keep API, business logic and AI orchestration separate.
- Use LangGraph only where stateful orchestration is useful.
- Avoid unnecessary abstractions and tiny modules.
- Avoid giant files and circular imports.
- Keep secrets outside source code.
- Use environment variables for configuration.
- Use database migrations.
- Write tests alongside implemented features.
- Never fabricate benchmark results or feature capabilities.
- Never silently switch from local inference to cloud inference.

## 5. Security and Privacy

- Treat uploaded documents and retrieved content as untrusted.
- Protect against prompt injection and indirect prompt injection.
- Enforce authorisation on the server.
- Isolate user conversations and documents.
- Never expose credentials or secrets to model context.
- Validate file uploads and API inputs.
- Apply timeouts and resource limits.
- Do not send local-mode data to cloud providers.
- Do not log sensitive prompts or credentials unnecessarily.

## 6. Development Workflow

- Work in small, testable milestones.
- Explain architectural decisions before implementing them.
- Never generate the entire application in one step.
- Do not install dependencies without explaining why.
- Do not create unnecessary files.
- Do not overwrite existing files without inspecting them.
- Run relevant tests after implementation.
- Report what was created, changed and tested.
- Ask for approval before moving to the next major milestone.

## 7. Current Milestone

Start with architecture and planning only.

Do not implement application code until the
product requirements, architecture, database design,
API contracts and project structure are reviewed.
