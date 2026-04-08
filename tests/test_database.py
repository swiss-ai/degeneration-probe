"""Tests for the database layer."""

import asyncio
import json
import pytest

from degeneration_probe.server.database import Database


@pytest.fixture
def db(tmp_path):
    """Create a temporary database."""
    db = Database(tmp_path / "test.db")
    asyncio.get_event_loop().run_until_complete(db.init())
    yield db
    asyncio.get_event_loop().run_until_complete(db.close())


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_create_session(db):
    session = run(db.create_session("localhost", 9000))
    assert session["worker_host"] == "localhost"
    assert session["worker_port"] == 9000
    assert session["status"] == "connected"


def test_get_current_session(db):
    run(db.create_session("localhost", 9000))
    session = run(db.get_current_session())
    assert session is not None
    assert session["worker_host"] == "localhost"


def test_get_current_session_when_none(db):
    session = run(db.get_current_session())
    assert session is None


def test_disconnect_session(db):
    session = run(db.create_session("localhost", 9000))
    run(db.disconnect_session(session["id"]))
    current = run(db.get_current_session())
    assert current is None


def test_save_generation(db):
    session = run(db.create_session("localhost", 9000))
    gen = run(db.save_generation(
        session_id=session["id"],
        prompt="Hello world",
        params={"temperature": 0.5},
        steering={"enabled": False},
        output_text="Hello world response",
        tokens=[{"token_text": "Hello", "probe_score": 0.1, "was_steered": False}],
        status="completed",
    ))
    assert gen["prompt"] == "Hello world"
    assert gen["status"] == "completed"


def test_list_generations(db):
    session = run(db.create_session("localhost", 9000))
    for i in range(3):
        run(db.save_generation(
            session_id=session["id"],
            prompt=f"Prompt {i}",
            params={},
            steering={},
            output_text=f"Response {i}",
            tokens=[],
            status="completed",
        ))
    gens = run(db.list_generations(limit=10, offset=0))
    assert len(gens) == 3


def test_get_generation_by_id(db):
    session = run(db.create_session("localhost", 9000))
    gen = run(db.save_generation(
        session_id=session["id"],
        prompt="Test",
        params={},
        steering={},
        output_text="Test response",
        tokens=[{"token_text": "t", "probe_score": 0.0, "was_steered": False}],
        status="completed",
    ))
    fetched = run(db.get_generation(gen["id"]))
    assert fetched is not None
    assert fetched["prompt"] == "Test"
    assert len(fetched["tokens"]) == 1
