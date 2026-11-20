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
    # Startup
    yield
    # Shutdown: dispose of database connection pool and checkpointer
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:5173"],
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
