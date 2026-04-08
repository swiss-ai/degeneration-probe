"""Integration tests for FastAPI routes."""

import pytest
from fastapi.testclient import TestClient

from degeneration_probe.server.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(db_path=tmp_path / "test.db")
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_strategies(client):
    resp = client.get("/api/strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert "temperature_boost" in data


def test_create_session(client):
    resp = client.post("/api/sessions", json={"worker_host": "localhost", "worker_port": 9000})
    assert resp.status_code == 200
    data = resp.json()
    assert data["worker_host"] == "localhost"
    assert data["status"] == "connected"


def test_get_current_session(client):
    client.post("/api/sessions", json={"worker_host": "localhost", "worker_port": 9000})
    resp = client.get("/api/sessions/current")
    assert resp.status_code == 200
    assert resp.json()["worker_host"] == "localhost"


def test_get_current_session_when_none(client):
    resp = client.get("/api/sessions/current")
    assert resp.status_code == 404


def test_delete_session(client):
    client.post("/api/sessions", json={"worker_host": "localhost", "worker_port": 9000})
    resp = client.delete("/api/sessions/current")
    assert resp.status_code == 200
    resp2 = client.get("/api/sessions/current")
    assert resp2.status_code == 404


def test_list_generations_empty(client):
    resp = client.get("/api/generations")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_generation_not_found(client):
    resp = client.get("/api/generations/999")
    assert resp.status_code == 404
