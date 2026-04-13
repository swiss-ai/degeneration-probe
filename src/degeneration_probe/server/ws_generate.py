"""WebSocket endpoint for streaming generation."""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import websockets

router = APIRouter()
log = logging.getLogger("backend.ws")


@router.websocket("/api/generate")
async def ws_generate(ws: WebSocket):
    """Relay generation between the Gradio client and the inference worker.

    1. Client sends a generate request (JSON with prompt, params, steering).
    2. Backend connects to the inference worker and forwards the request.
    3. Backend streams token messages from worker back to client.
    4. If the client disconnects, the worker connection is closed too.
    """
    await ws.accept()
    db = ws.app.state.db
    worker_ws = None

    try:
        raw = await ws.receive_text()
        request = json.loads(raw)
        log.info("Generate request: prompt=%d chars", len(request.get("prompt", "")))

        session = await db.get_current_session()
        if session is None:
            log.warning("No worker session — rejecting request")
            await ws.send_json({"type": "error", "message": "No worker connected"})
            await ws.close()
            return

        worker_url = f"ws://{session['worker_host']}:{session['worker_port']}"
        log.info("Connecting to worker at %s", worker_url)

        tokens = []
        output_text = ""
        worker_ws = await websockets.connect(worker_url, ping_timeout=None)
        log.info("Connected to worker")
        await worker_ws.send(json.dumps(request))

        async for msg in worker_ws:
            data = json.loads(msg)

            # Try to send to client — if client disconnected, break out
            try:
                await ws.send_json(data)
            except (WebSocketDisconnect, RuntimeError):
                log.warning("UI client disconnected at token %d", len(tokens))
                break

            if data["type"] == "token":
                tokens.append({
                    "token_id": data["token_id"],
                    "token_text": data["token_text"],
                    "position": data["position"],
                    "probe_score": data["probe_score"],
                    "was_steered": data["was_steered"],
                })
                output_text += data["token_text"]
            elif data["type"] == "done":
                log.info("Generation done: %d tokens", len(tokens))
                break

        # Save whatever we generated (even if stopped early)
        if tokens:
            await db.save_generation(
                session_id=session["id"],
                prompt=request.get("prompt", ""),
                params=request.get("params", {}),
                steering=request.get("steering", {}),
                output_text=output_text,
                tokens=tokens,
                status="completed",
            )
            log.info("Saved generation: %d tokens", len(tokens))

    except WebSocketDisconnect:
        log.info("UI client disconnected (tokens so far: %d)", len(tokens) if 'tokens' in dir() else 0)
    except Exception as e:
        log.exception("Error in ws_generate")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if worker_ws is not None:
            try:
                await worker_ws.close()
            except Exception:
                pass
