"""WebSocket endpoint for streaming generation."""

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
    4. Client can send 'stop' or 'update_steering' mid-stream.
    """
    await ws.accept()
    db = ws.app.state.db

    try:
        # Wait for the generation request from the client
        raw = await ws.receive_text()
        request = json.loads(raw)

        # Get current worker session
        session = await db.get_current_session()
        if session is None:
            await ws.send_json({"type": "error", "message": "No worker connected"})
            await ws.close()
            return

        worker_url = f"ws://{session['worker_host']}:{session['worker_port']}"

        # Connect to worker and relay
        tokens = []
        output_text = ""
        async with websockets.connect(worker_url) as worker_ws:
            await worker_ws.send(json.dumps(request))

            async for msg in worker_ws:
                data = json.loads(msg)
                await ws.send_json(data)

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

        # Save to database
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
