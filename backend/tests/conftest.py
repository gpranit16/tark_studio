"""Pytest configuration and shared async test fixtures."""

import os
import sys
import pytest
from httpx import AsyncClient, ASGITransport

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.config import settings
from app.database import engine
from app.ai.graph import close_checkpointer


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
async def cleanup_pools():
    yield
    await close_checkpointer()
    await engine.dispose()


@pytest.fixture
async def test_client():
    """Async HTTP client fixture configured for ASGI transport."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
