"""Entrypoint runner for TARK Studio backend.

Ensures WindowsSelectorEventLoopPolicy is configured before Uvicorn starts the event loop,
which is required by Psycopg async on Windows.
"""

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug and sys.platform == "win32",
    )
