"""Entrypoint runner for TARK Studio backend.

Ensures WindowsSelectorEventLoopPolicy is configured before Uvicorn starts the event loop,
which is required by Psycopg async on Windows.
"""

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
