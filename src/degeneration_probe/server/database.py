"""SQLite database for session and generation storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite


class Database:
    """Async SQLite database wrapper for sessions and generations."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None

    async def init(self):
        """Initialize the database and create tables."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                worker_host TEXT NOT NULL,
                worker_port INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'connected'
            );
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                prompt TEXT NOT NULL,
                params_json TEXT NOT NULL,
                steering_json TEXT NOT NULL,
                output_text TEXT NOT NULL DEFAULT '',
                tokens_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'running',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    async def create_session(self, worker_host: str, worker_port: int) -> dict:
        """Create a new session and mark any existing ones as disconnected."""
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE sessions SET status = 'disconnected' WHERE status = 'connected'"
        )
        cursor = await self._db.execute(
            "INSERT INTO sessions (created_at, worker_host, worker_port, status) VALUES (?, ?, ?, 'connected')",
            (now, worker_host, worker_port),
        )
        await self._db.commit()
        row = await self._db.execute_fetchall(
            "SELECT * FROM sessions WHERE id = ?", (cursor.lastrowid,)
        )
        return dict(row[0])

    async def get_current_session(self) -> dict | None:
        """Get the currently connected session, or None."""
        rows = await self._db.execute_fetchall(
            "SELECT * FROM sessions WHERE status = 'connected' ORDER BY id DESC LIMIT 1"
        )
        return dict(rows[0]) if rows else None

    async def disconnect_session(self, session_id: int):
        """Mark a session as disconnected."""
        await self._db.execute(
            "UPDATE sessions SET status = 'disconnected' WHERE id = ?", (session_id,)
        )
        await self._db.commit()

    async def save_generation(
        self,
        session_id: int,
        prompt: str,
        params: dict,
        steering: dict,
        output_text: str,
        tokens: list[dict],
        status: str,
    ) -> dict:
        """Save a generation record."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            """INSERT INTO generations
               (session_id, prompt, params_json, steering_json, output_text, tokens_json, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                prompt,
                json.dumps(params),
                json.dumps(steering),
                output_text,
                json.dumps(tokens),
                status,
                now,
            ),
        )
        await self._db.commit()
        row = await self._db.execute_fetchall(
            "SELECT * FROM generations WHERE id = ?", (cursor.lastrowid,)
        )
        result = dict(row[0])
        result["tokens"] = json.loads(result.pop("tokens_json"))
        result["params"] = json.loads(result.pop("params_json"))
        result["steering"] = json.loads(result.pop("steering_json"))
        return result

    async def list_generations(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """List generations, most recent first."""
        rows = await self._db.execute_fetchall(
            "SELECT id, session_id, prompt, status, created_at FROM generations ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(r) for r in rows]

    async def get_generation(self, gen_id: int) -> dict | None:
        """Get a single generation with full token data."""
        rows = await self._db.execute_fetchall(
            "SELECT * FROM generations WHERE id = ?", (gen_id,)
        )
        if not rows:
            return None
        result = dict(rows[0])
        result["tokens"] = json.loads(result.pop("tokens_json"))
        result["params"] = json.loads(result.pop("params_json"))
        result["steering"] = json.loads(result.pop("steering_json"))
        return result
