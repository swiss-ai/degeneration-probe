"""Integration tests for FastAPI routes."""

import pytest
from fastapi.testclient import TestClient

from degeneration_probe.server.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(db_path=tmp_path / "test.db")
    with TestClient(app) as c:
        yield c


def test_create_session(client):
    resp = client.post("/api/sessions", json={"worker_host": "localhost", "worker_port": 9000})
    assert resp.status_code == 200
    data = resp.json()
    assert data["worker_host"] == "localhost"
    assert data["status"] == "connected"


def test_worker_info_without_session(client):
    resp = client.get("/api/worker/info")
    assert resp.status_code == 404
