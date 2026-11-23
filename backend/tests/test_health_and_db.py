"""Unit and integration tests for API health and database connectivity configuration."""

import pytest
from httpx import AsyncClient
from app.config import Settings
from app.database import check_db_connectivity


@pytest.mark.asyncio
async def test_root_endpoint(test_client: AsyncClient):
    """Verify that root endpoint returns service identity and EMBER SAGE design branding."""
    response = await test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "TARK Studio API"
    assert data["status"] == "online"
    assert data["design_identity"] == "EMBER SAGE"


@pytest.mark.asyncio
async def test_health_endpoint_structure(test_client: AsyncClient):
    """Verify that health check endpoint returns structured database status."""
    response = await test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "connected" in data["database"]
    assert "has_vector_extension" in data["database"]


def test_database_url_resolution():
    """Verify that Settings constructs asyncpg connection string dynamically when DATABASE_URL is not set."""
    custom_settings = Settings(
        DATABASE_URL=None,
        POSTGRES_USER="tark_user",
        POSTGRES_PASSWORD="secure_password",
        POSTGRES_HOST="10.0.0.5",
        POSTGRES_PORT=5433,
        POSTGRES_DB="tark_prod",
    )
    expected = "postgresql+asyncpg://tark_user:secure_password@10.0.0.5:5433/tark_prod"
    assert custom_settings.async_database_url == expected


@pytest.mark.asyncio
async def test_models_endpoint(test_client: AsyncClient):
    """Verify that models endpoint returns available models list and default."""
    response = await test_client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0
    assert data["default"] == "openai/gpt-oss-120b"


@pytest.mark.asyncio
async def test_chat_stream_payload_validation(test_client: AsyncClient):
    """Verify that chat stream endpoint validates missing message payloads."""
    response = await test_client.post("/api/v1/chat/stream", json={"messages": []})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_private_network_access_cors_header(test_client: AsyncClient):
    """Verify Chrome Local Network Access (LNA) preflight header response."""
    response = await test_client.options(
        "/api/v1/health",
        headers={
            "Origin": "https://tark-studio.pages.dev",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-private-network") == "true"
    assert response.headers.get("access-control-allow-origin") == "https://tark-studio.pages.dev"
