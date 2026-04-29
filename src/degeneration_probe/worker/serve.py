"""Inference worker: WebSocket server that runs generation on GPU."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import websockets
from websockets.exceptions import ConnectionClosed

from degeneration_probe._fork_path import ensure_fork_on_syspath
from degeneration_probe.model_utils import load_model_and_tokenizer, resolve_torch_dtype
from degeneration_probe.worker.engine import GenerationEngine
from degeneration_probe.worker.steering import get_strategy

log = logging.getLogger("worker")


def handle_message(raw: str) -> dict:
    """Parse and validate an incoming WebSocket message."""
    return json.loads(raw)


async def run_generation(ws, engine: GenerationEngine, request: dict):
    """Run generation and stream token results back over WebSocket."""
    prompt = request["prompt"]
    params = request.get("params", {})
    steering_cfg = request.get("steering", {})

    log.info("Generate: prompt=%d chars, max_new_tokens=%s, temperature=%s",
             len(prompt), params.get("max_new_tokens", 4096), params.get("temperature", 0.01))

    # Build steering strategy if enabled
    steering = None
    threshold = 0.8
    if steering_cfg.get("enabled"):
        strategy_name = steering_cfg.get("strategy", "temperature_boost")
        strategy_params = {
            k: v for k, v in steering_cfg.items()
            if k not in ("enabled", "strategy", "threshold")
        }
        steering = get_strategy(strategy_name, **strategy_params)
        threshold = steering_cfg.get("threshold", 0.8)
        log.info("Steering enabled: strategy=%s, threshold=%.2f", strategy_name, threshold)

    total = 0
    batch: list[dict] = []
    batch_size = int(params.get("batch_size", 4))
    t0 = time.monotonic()

    async def flush_batch() -> bool:
        """Send the current batch. Returns False if the client disconnected."""
        if not batch:
            return True
        try:
            await ws.send(json.dumps({"type": "tokens", "tokens": batch}))
        except ConnectionClosed:
            log.warning("Client disconnected at token %d", total)
            engine.request_stop()
            return False
        batch.clear()
        return True

    # Run the sync PyTorch generation loop in a worker thread and funnel
    # TokenResults back through an asyncio.Queue. This keeps the event loop
    # free to process incoming control messages (update_params, stop) in
    # real time.
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    _SENTINEL = object()

    def producer() -> None:
        try:
            for r in engine.generate(
                prompt=prompt,
                max_new_tokens=params.get("max_new_tokens", 4096),
                temperature=params.get("temperature", 0.01),
                top_p=params.get("top_p", 0.9),
                steering=steering,
                steering_threshold=threshold,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(r), loop).result()
        except BaseException as exc:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            return
        asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop).result()

    producer_task = loop.run_in_executor(None, producer)

    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            r = item
            batch.append({
                "token_id": r.token_id,
                "token_text": r.token_text,
                "position": r.position,
                "probe_score": r.probe_score,
                "was_steered": r.was_steered,
            })
            total += 1
            if len(batch) >= batch_size:
                if not await flush_batch():
                    return

        if not await flush_batch():
            return

        elapsed = time.monotonic() - t0
        log.info("Generation complete: %d tokens in %.1fs (%.1f tok/s)",
                 total, elapsed, total / elapsed if elapsed > 0 else 0)
        await ws.send(json.dumps({
            "type": "done",
            "total_tokens": total,
        }))
    except ConnectionClosed:
        log.warning("Connection closed during generation at token %d", total)
        engine.request_stop()
    except Exception:
        log.exception("Error during generation at token %d", total)
        engine.request_stop()
    finally:
        # Wait for the producer thread to wind down so we don't leak it.
        try:
            await producer_task
        except Exception:
            log.exception("Producer thread finished with error")


async def handler(ws, engine: GenerationEngine):
    """Handle a single WebSocket connection.

    The handler multiplexes incoming messages: `generate` kicks off a
    generation task and `update_params` / `stop` mutate the running engine
    so the UI sliders can retune temperature, top_p, and steering threshold
    mid-stream.
    """
    remote = ws.remote_address
    log.info("Client connected: %s", remote)
    current_gen: Optional[asyncio.Task] = None
    try:
        async for raw in ws:
            msg = handle_message(raw)
            action = msg.get("action")

            if action == "generate":
                if current_gen is not None and not current_gen.done():
                    engine.request_stop()
                    try:
                        await current_gen
                    except Exception:
                        log.exception("Previous generation errored while being preempted")
                current_gen = asyncio.create_task(run_generation(ws, engine, msg))
            elif action == "update_params":
                params = msg.get("params", {})
                log.info("update_params received: %s", params)
                engine.update_live_params(**params)
            elif action == "info":
                await ws.send(json.dumps({
                    "type": "info",
                    "model": engine.model_name,
                    "has_probe": engine.probe is not None,
                }))
            elif action == "stop":
                log.info("Stop requested by client")
                engine.request_stop()
    finally:
        if current_gen is not None and not current_gen.done():
            engine.request_stop()
            try:
                await current_gen
            except Exception:
                pass
        log.info("Client disconnected: %s", remote)


async def start_server(host: str, port: int, engine: GenerationEngine):
    """Start the WebSocket server."""
    async def ws_handler(ws):
        await handler(ws, engine)

    server = await websockets.serve(ws_handler, host, port, ping_timeout=None)
    log.info("Inference worker listening on ws://%s:%d", host, port)
    await server.wait_closed()


def _resolve_model_name(args_model: str | None, probe_path: str | None) -> str:
    """Pick the right model to load.

    Rules:
      - no probe + no --model           → error.
      - no probe + --model M            → load M.
      - probe present                   → read its degeneration_meta.json
                                          for model_name; that's the model.
                                          If --model is also given and matches,
                                          fine; if it disagrees, error rather
                                          than silently load the wrong base.
    """
    probe_model: str | None = None
    if probe_path:
        meta_path = Path(probe_path) / "degeneration_meta.json"
        if not meta_path.exists():
            raise SystemExit(
                f"Probe checkpoint at {probe_path} is missing degeneration_meta.json — "
                f"can't determine which model it was trained on."
            )
        meta = json.loads(meta_path.read_text())
        probe_model = meta.get("model_name")
        if not probe_model:
            raise SystemExit(
                f"degeneration_meta.json at {meta_path} has no 'model_name' field."
            )

    if probe_model is not None:
        if args_model and args_model != probe_model:
            raise SystemExit(
                f"--model {args_model!r} disagrees with the probe's trained model "
                f"{probe_model!r}. Loading the probe on a different base would give "
                f"meaningless scores. Either drop --model or pass --model {probe_model!r}."
            )
        return probe_model

    if not args_model:
        raise SystemExit("Must pass either --model or --probe (with a checkpoint that names its model).")
    return args_model


def main():
    parser = argparse.ArgumentParser(description="Start inference worker")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model name (optional if --probe is set; the probe's meta names its model)")
    parser.add_argument("--probe", type=str, default=None, help="Path to saved probe checkpoint")
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype (bfloat16, float32, etc.)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    model_name = _resolve_model_name(args.model, args.probe)
    print(f"Loading model {model_name}...")
    dtype = resolve_torch_dtype(args.dtype)
    model, tokenizer = load_model_and_tokenizer(model_name, torch_dtype=dtype)

    probe = None
    if args.probe:
        print(f"Loading probe from {args.probe}...")
        ensure_fork_on_syspath()
        from degeneration.probe_loader import load_probe
        probe = load_probe(args.probe, model)

    engine = GenerationEngine(model=model, tokenizer=tokenizer, probe=probe, model_name=model_name)
    print("Model loaded. Starting server...")
    asyncio.run(start_server(args.host, args.port, engine))


if __name__ == "__main__":
    main()
