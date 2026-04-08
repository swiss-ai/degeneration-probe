"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from degeneration_probe.server.database import Database
from degeneration_probe.server.routes_health import router as health_router
from degeneration_probe.server.routes_sessions import router as sessions_router
from degeneration_probe.server.routes_generations import router as generations_router
from degeneration_probe.server.ws_generate import router as ws_router

DEFAULT_DB_PATH = Path("data/degeneration_probe.db")


def create_app(db_path: str | Path = DEFAULT_DB_PATH) -> FastAPI:
    """Create and configure the FastAPI application."""
    db = Database(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db.init()
        app.state.db = db
        yield
        await db.close()

    app = FastAPI(
        title="Degeneration Probe API",
        description="Backend for real-time LLM degeneration detection and steering",
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(sessions_router)
    app.include_router(generations_router)
    app.include_router(ws_router)

    return app
