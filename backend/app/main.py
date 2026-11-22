"""TARK Studio — FastAPI Main Application Entrypoint."""

import sys
import asyncio

# On Windows, Psycopg async requires the SelectorEventLoopPolicy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, check_db_connectivity
from app.ai.graph import close_checkpointer
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context: initialize and cleanup resources."""
    # Startup: Warm up ONNX reranker ahead of time to eliminate user-perceived cold-start latency
    try:
        from app.rag.reranker import get_reranker
        import logging
        import time
        log = logging.getLogger("app.startup")
        t0 = time.perf_counter()
        log.info("Warming up ONNX cross-encoder model during application startup...")
        reranker = get_reranker("onnx")
        # Run small dummy inference to prime CPU weights and memory buffers
        await asyncio.to_thread(reranker.rerank, "warmup query", [{"content": "warmup candidate chunk"}], 1)
        log.info(f"ONNX reranker warmed up successfully in {round(time.perf_counter() - t0, 3)}s.")
    except Exception as exc:
        import logging
        logging.getLogger("app.startup").warning(f"ONNX reranker warmup skipped/deferred: {exc}")

    # Startup: Warm up PostgreSQL checkpointer connection and schema
    try:
        from app.ai.graph import get_checkpointer, get_compiled_graph
        import logging
        log = logging.getLogger("app.startup")
        log.info("Warming up PostgreSQL checkpointer connection...")
        await get_checkpointer()
        log.info("Warming up compiled StateGraph...")
        await get_compiled_graph()
        log.info("PostgreSQL checkpointer and StateGraph warmed up successfully.")
    except Exception as exc:
        import logging
        logging.getLogger("app.startup").warning(f"Checkpointer warmup deferred: {exc}")

    # Startup: Ensure embedding provider model is available
    try:
        from app.rag.embeddings import embeddings_client
        import logging
        log = logging.getLogger("app.startup")
        log.info(f"Verifying embedding model availability ({embeddings_client.model_name})...")
        asyncio.create_task(embeddings_client.ensure_model_available())
    except Exception as exc:
        import logging
        logging.getLogger("app.startup").warning(f"Embedding model check deferred: {exc}")

    yield
    # Shutdown: gracefully await pending background rolling summaries, then dispose of checkpointer & DB
    try:
        from app.ai.memory import background_summary_manager
        import logging
        await background_summary_manager.wait_all_pending(timeout=10.0)
    except Exception as exc:
        import logging
        logging.getLogger("app.shutdown").warning(f"Error waiting for pending summaries during shutdown: {exc}")

    await close_checkpointer()
    await engine.dispose()


app = FastAPI(
    title="TARK Studio API",
    version="0.1.0",
    description="Local-first AI Workspace backend service orchestrated with LangGraph.",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# Configure CORS for local development and production Cloudflare Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_cors_origins,
    allow_origin_regex=r"^https:\/\/.*\.pages\.dev$" if not settings.debug else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/", tags=["Root"])
async def root():
    """Root health check returning service identity."""
    return {
        "service": "TARK Studio API",
        "version": "0.1.0",
        "status": "online",
        "design_identity": "EMBER SAGE",
        "orchestration": "LangGraph",
    }


@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    """System, database, and provider connectivity status."""
    db_status = await check_db_connectivity()
    return {
        "status": "healthy" if db_status["connected"] else "degraded",
        "app_env": settings.app_env,
        "database": db_status,
        "providers": {
            "groq_configured": bool(settings.groq_api_key),
            "gemini_configured": bool(settings.gemini_api_key),
            "ollama_base_url": settings.ollama_base_url,
        },
    }
