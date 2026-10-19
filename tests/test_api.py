"""API tests — require a running PostgreSQL + pgvector instance.
Run with: pytest tests/test_api.py -v (after `docker compose up -d`)
"""
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client():
    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_stats(client):
    response = await client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "sources" in data
