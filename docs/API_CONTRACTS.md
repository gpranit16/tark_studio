# TARK Studio — API Contracts & Wire Specification

**Version:** 1.0.0  
**Status:** Architecture & Planning Phase (Awaiting Review)  
**Base URL:** `/api/v1`  
**Communication Protocols:** REST (JSON) & Server-Sent Events (SSE)

---

## 1. General Principles & Error Standards

### 1.1 Content Negotiation & Encoding
- All standard REST request and response payloads use `application/json` encoded in UTF-8.
- Streaming endpoints use `text/event-stream` with `Cache-Control: no-cache` and `Connection: keep-alive`.
- JSON keys use `snake_case` across all requests, responses, and SSE event payloads.
- Date and time fields are ISO 8601 strings with timezone offsets (e.g., `2026-11-18T10:00:00.000Z`).

### 1.2 Unified Error Response Format
All error responses return a standardized RFC 7807-inspired JSON structure:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Conversation with id '8a9f...3b1' does not exist.",
    "details": {},
    "timestamp": "2026-11-18T10:05:00.000Z"
  }
}
```

#### Standard Error Codes:
| HTTP Status | Error Code | Description |
| :--- | :--- | :--- |
| `400` | `INVALID_REQUEST` | Malformed JSON or request parameter violation |
| `403` | `PRIVACY_VIOLATION` | Attempted cloud routing on a local-only conversation |
| `404` | `RESOURCE_NOT_FOUND` | Specified conversation, document, or memory not found |
| `422` | `UNPROCESSABLE_ENTITY` | Pydantic payload schema validation failure |
| `502` | `PROVIDER_UNAVAILABLE` | Target inference engine (Ollama, Groq, Gemini) unreachable |
| `504` | `INFERENCE_TIMEOUT` | Upstream model response timed out |

---

## 2. Conversation Management Endpoints

### 2.1 List Conversations
- **Method & Path:** `GET /api/v1/conversations`
- **Query Parameters:**
  - `limit` (integer, default: 50, max: 100)
  - `offset` (integer, default: 0)
  - `is_archived` (boolean, optional, default: false)
  - `search` (string, optional - searches titles)
- **Response `200 OK`:**
```json
{
  "items": [
    {
      "id": "e6a4b123-8c4b-4a57-9dbf-1c4b8109d901",
      "title": "PostgreSQL HNSW Index Optimization",
      "model_id": "llama3.2",
      "provider": "ollama",
      "is_local_only": true,
      "is_archived": false,
      "message_count": 8,
      "created_at": "2026-11-18T09:30:00.000Z",
      "updated_at": "2026-11-18T09:45:00.000Z"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### 2.2 Create Conversation
- **Method & Path:** `POST /api/v1/conversations`
- **Request Body:**
```json
{
  "title": "Async SQLAlchemy Architecture",
  "model_id": "llama-3.3-70b-versatile",
  "provider": "groq",
  "system_prompt": "You are an expert Python systems architect.",
  "is_local_only": false
}
```
- **Response `201 Created`:**
```json
{
  "id": "e6a4b123-8c4b-4a57-9dbf-1c4b8109d901",
  "title": "Async SQLAlchemy Architecture",
  "model_id": "llama-3.3-70b-versatile",
  "provider": "groq",
  "system_prompt": "You are an expert Python systems architect.",
  "is_local_only": false,
  "is_archived": false,
  "created_at": "2026-11-18T10:00:00.000Z",
  "updated_at": "2026-11-18T10:00:00.000Z"
}
```

### 2.3 Get Conversation Details
- **Method & Path:** `GET /api/v1/conversations/{conversation_id}`
- **Response `200 OK`:**
```json
{
  "id": "e6a4b123-8c4b-4a57-9dbf-1c4b8109d901",
  "title": "Async SQLAlchemy Architecture",
  "model_id": "llama-3.3-70b-versatile",
  "provider": "groq",
  "system_prompt": "You are an expert Python systems architect.",
  "is_local_only": false,
  "is_archived": false,
  "attached_documents": [
    {
      "id": "2bc99478-f712-45e0-8fa4-1e05d2146fef",
      "name": "sqlalchemy_2_guide.md"
    }
  ],
  "created_at": "2026-11-18T10:00:00.000Z",
  "updated_at": "2026-11-18T10:00:00.000Z"
}
```

### 2.4 Update Conversation
- **Method & Path:** `PATCH /api/v1/conversations/{conversation_id}`
- **Request Body (all fields optional):**
```json
{
  "title": "Updated Title",
  "model_id": "gemini-1.5-pro",
  "provider": "gemini",
  "system_prompt": "Updated system instructions",
  "is_archived": false
}
```
- **Response `200 OK`:** Returns updated conversation object.

### 2.5 Delete Conversation
- **Method & Path:** `DELETE /api/v1/conversations/{conversation_id}`
- **Response `204 No Content`**

---

## 3. Messages & Real-Time Streaming Endpoints

### 3.1 Send Message & Stream Response (Fetch + ReadableStream SSE)
- **Method & Path:** `POST /api/v1/conversations/{conversation_id}/messages`
- **Client Protocol:** Client dispatches request via `fetch()` with JSON payload, passes `AbortController.signal`, and consumes the stream using `response.body.getReader()`.
- **Request Body:**
```json
{
  "content": "Explain how HNSW indexing compares to IVFFlat in pgvector.",
  "model_id": "llama3.2",
  "provider": "ollama",
  "temperature": 0.7,
  "enable_rag": true,
  "enable_memory": true
}
```
- **Response `200 OK`:**  
  `Content-Type: text/event-stream`  
  `Transfer-Encoding: chunked`  
  `X-Accel-Buffering: no`

#### Wire Event Sequence Specification:
```
event: meta
data: {"sequence_id": 1, "message_id": "c138fd94-f201-4475-a08e-173693247063", "conversation_id": "e6a4b123-8c4b-4a57-9dbf-1c4b8109d901", "provider": "ollama", "model": "llama3.2"}

event: memory_applied
data: {"sequence_id": 2, "id": "5fa23d11-5364-42f0-936d-97996c56134b", "category": "preference", "content": "User prefers concise technical comparisons with markdown tables."}

event: citation
data: {"sequence_id": 3, "document_id": "2bc99478-f712-45e0-8fa4-1e05d2146fef", "chunk_id": "b304c2ae-081e-4509-913a-a111b1fe4e71", "document_name": "pgvector_notes.md", "snippet": "HNSW builds a multi-layer graph structure providing high recall with low query latency...", "similarity": 0.88}

event: token
data: {"sequence_id": 4, "delta": "HNSW"}

event: token
data: {"sequence_id": 5, "delta": " (Hierarchical"}

event: token
data: {"sequence_id": 6, "delta": " Navigable"}

event: token
data: {"sequence_id": 7, "delta": " Small World)"}

event: meta
data: {"sequence_id": 8, "prompt_tokens": 128, "completion_tokens": 64, "ttft_ms": 195, "total_duration_ms": 840, "finish_reason": "stop"}

event: done
data: {"sequence_id": 9, "status": "completed"}
```

#### Disconnect, Cancellation & Duplicate Prevention:
- **Client Cancellation:** Aborting the client request via `AbortController` triggers immediate backend generator cancellation and frees LLM resources.
- **Duplicate Prevention:** Dispatches with the same turn state while generation is already in flight for `conversation_id` return `409 Conflict`.
- **Explicit Reconnect:** If the TCP connection is severed prior to `event: done`, the frontend does not fall back to browser `EventSource` auto-retry; instead, it offers explicit turn retry/resume using the last acknowledged `sequence_id`.

### 3.2 List Messages in Conversation
- **Method & Path:** `GET /api/v1/conversations/{conversation_id}/messages`
- **Query Parameters:**
  - `limit` (integer, default: 100)
  - `before_id` (UUID, optional for backwards pagination)
- **Response `200 OK`:**
```json
{
  "items": [
    {
      "id": "c138fd94-f201-4475-a08e-173693247063",
      "conversation_id": "e6a4b123-8c4b-4a57-9dbf-1c4b8109d901",
      "role": "user",
      "content": "Explain how HNSW indexing compares to IVFFlat in pgvector.",
      "created_at": "2026-11-18T10:01:00.000Z"
    },
    {
      "id": "8f03126b-d34e-4e68-bc25-2ff175d0505b",
      "conversation_id": "e6a4b123-8c4b-4a57-9dbf-1c4b8109d901",
      "role": "assistant",
      "content": "HNSW builds a multi-layer graph...",
      "model": "llama3.2",
      "provider": "ollama",
      "prompt_tokens": 128,
      "completion_tokens": 64,
      "latency_ms": 840,
      "finish_reason": "stop",
      "metadata": {
        "ttft_ms": 195,
        "citations": [
          {
            "document_name": "pgvector_notes.md",
            "snippet": "HNSW builds a multi-layer graph...",
            "similarity": 0.88
          }
        ]
      },
      "created_at": "2026-11-18T10:01:01.000Z"
    }
  ],
  "total": 2
}
```

---

## 4. Models & Provider Management Endpoints

### 4.1 List Available Models
- **Method & Path:** `GET /api/v1/models`
- **Response `200 OK`:**
```json
{
  "models": [
    {
      "id": "llama3.2:latest",
      "name": "Llama 3.2 (3B)",
      "provider": "ollama",
      "is_local": true,
      "context_window": 131072,
      "status": "online"
    },
    {
      "id": "llama-3.3-70b-versatile",
      "name": "Llama 3.3 (70B Versatile)",
      "provider": "groq",
      "is_local": false,
      "context_window": 128000,
      "status": "online"
    },
    {
      "id": "gemini-1.5-pro",
      "name": "Gemini 1.5 Pro",
      "provider": "gemini",
      "is_local": false,
      "context_window": 2097152,
      "status": "online"
    }
  ]
}
```

### 4.2 Provider Health & Status
- **Method & Path:** `GET /api/v1/providers/health`
- **Response `200 OK`:**
```json
{
  "ollama": {
    "status": "online",
    "endpoint": "http://localhost:11434",
    "version": "0.5.4",
    "models_loaded": ["llama3.2:latest"]
  },
  "groq": {
    "status": "configured",
    "authenticated": true
  },
  "gemini": {
    "status": "configured",
    "authenticated": true
  }
}
```

---

## 5. Document Management & RAG Endpoints

### 5.1 Upload Document
- **Method & Path:** `POST /api/v1/documents/upload`
- **Content-Type:** `multipart/form-data`
- **Form Fields:**
  - `file`: Binary file (PDF, TXT, MD, Code)
  - `conversation_id`: Optional UUID (if attached immediately to an open thread)
- **Response `201 Created`:**
```json
{
  "id": "2bc99478-f712-45e0-8fa4-1e05d2146fef",
  "name": "sqlalchemy_2_guide.md",
  "file_size": 48210,
  "mime_type": "text/markdown",
  "sha256_hash": "a1b2c3d4...",
  "status": "processing",
  "total_chunks": 0,
  "created_at": "2026-11-18T10:03:00.000Z"
}
```

### 5.2 List Documents
- **Method & Path:** `GET /api/v1/documents`
- **Query Parameters:**
  - `status` (optional: `pending`, `processing`, `indexed`, `failed`)
- **Response `200 OK`:**
```json
{
  "items": [
    {
      "id": "2bc99478-f712-45e0-8fa4-1e05d2146fef",
      "name": "sqlalchemy_2_guide.md",
      "file_size": 48210,
      "mime_type": "text/markdown",
      "status": "indexed",
      "total_chunks": 14,
      "created_at": "2026-11-18T10:03:00.000Z",
      "updated_at": "2026-11-18T10:03:02.000Z"
    }
  ],
  "total": 1
}
```

### 5.3 Delete Document
- **Method & Path:** `DELETE /api/v1/documents/{document_id}`
- **Response `204 No Content`** (cascades to all chunks and junction records).

### 5.4 Attach Document to Conversation
- **Method & Path:** `POST /api/v1/conversations/{conversation_id}/documents`
- **Request Body:**
```json
{
  "document_id": "2bc99478-f712-45e0-8fa4-1e05d2146fef"
}
```
- **Response `201 Created`**

### 5.5 Detach Document from Conversation
- **Method & Path:** `DELETE /api/v1/conversations/{conversation_id}/documents/{document_id}`
- **Response `204 No Content`**

---

## 6. Memory Management Endpoints

### 6.1 List Memories
- **Method & Path:** `GET /api/v1/memories`
- **Query Parameters:**
  - `category` (optional: `preference`, `fact`, `constraint`, `instruction`)
  - `is_active` (optional boolean, default: true)
- **Response `200 OK`:**
```json
{
  "items": [
    {
      "id": "5fa23d11-5364-42f0-936d-97996c56134b",
      "category": "preference",
      "content": "User prefers concise technical comparisons with markdown tables.",
      "confidence": 0.95,
      "access_count": 4,
      "is_active": true,
      "source_conversation_id": "e6a4b123-8c4b-4a57-9dbf-1c4b8109d901",
      "last_accessed_at": "2026-11-18T10:01:00.000Z",
      "created_at": "2026-11-17T18:20:00.000Z"
    }
  ],
  "total": 1
}
```

### 6.2 Create Explicit Memory
- **Method & Path:** `POST /api/v1/memories`
- **Request Body:**
```json
{
  "category": "constraint",
  "content": "Never write raw SQL strings when SQLAlchemy Core or ORM constructs can express the query."
}
```
- **Response `201 Created`**

### 6.3 Update Memory
- **Method & Path:** `PATCH /api/v1/memories/{memory_id}`
- **Request Body (all optional):**
```json
{
  "content": "Updated constraint description...",
  "category": "constraint",
  "is_active": false
}
```
- **Response `200 OK`**

### 6.4 Delete Memory
- **Method & Path:** `DELETE /api/v1/memories/{memory_id}`
- **Response `204 No Content`**
