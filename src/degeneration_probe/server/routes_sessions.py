"""Session management endpoints."""

import json

import websockets
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class CreateSessionRequest(BaseModel):
    worker_host: str
    worker_port: int


@router.post("/api/sessions")
async def create_session(req: CreateSessionRequest, request: Request):
    db = request.app.state.db
    session = await db.create_session(req.worker_host, req.worker_port)
    return session


@router.get("/api/sessions/current")
async def get_current_session(request: Request):
    db = request.app.state.db
    session = await db.get_current_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No active session")
    return session


@router.delete("/api/sessions/current")
async def delete_session(request: Request):
    db = request.app.state.db
    session = await db.get_current_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No active session")
    await db.disconnect_session(session["id"])
    return {"status": "disconnected"}


@router.get("/api/worker/info")
async def worker_info(request: Request):
    """Ask the current worker what model it's serving."""
    db = request.app.state.db
    session = await db.get_current_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No active session")
    worker_url = f"ws://{session['worker_host']}:{session['worker_port']}"
    try:
        async with websockets.connect(worker_url, ping_timeout=None, open_timeout=5) as ws:
            await ws.send(json.dumps({"action": "info"}))
            raw = await ws.recv()
            return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Worker unreachable: {e}")
