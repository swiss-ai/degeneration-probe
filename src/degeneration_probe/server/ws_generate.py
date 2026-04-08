"""WebSocket endpoint for streaming generation."""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import websockets

router = APIRouter()


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

        session = await db.get_current_session()
        if session is None:
            await ws.send_json({"type": "error", "message": "No worker connected"})
            await ws.close()
            return

        worker_url = f"ws://{session['worker_host']}:{session['worker_port']}"

        tokens = []
        output_text = ""
        worker_ws = await websockets.connect(worker_url)
        await worker_ws.send(json.dumps(request))

        async for msg in worker_ws:
            data = json.loads(msg)

            # Try to send to client — if client disconnected, break out
            try:
                await ws.send_json(data)
            except (WebSocketDisconnect, RuntimeError):
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

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Always close the worker connection to stop generation
        if worker_ws is not None:
            try:
                await worker_ws.close()
            except Exception:
                pass
