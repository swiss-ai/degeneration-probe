"""Inference worker: WebSocket server that runs generation on GPU."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch
import websockets
from websockets.exceptions import ConnectionClosed

from degeneration_probe.model_utils import load_model_and_tokenizer, resolve_torch_dtype
from degeneration_probe.probe import SequenceProbe
from degeneration_probe.worker.engine import GenerationEngine
from degeneration_probe.worker.steering import get_strategy


def handle_message(raw: str) -> dict:
    """Parse and validate an incoming WebSocket message."""
    return json.loads(raw)


async def run_generation(ws, engine: GenerationEngine, request: dict):
    """Run generation and stream token results back over WebSocket."""
    prompt = request["prompt"]
    params = request.get("params", {})
    steering_cfg = request.get("steering", {})

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

    total = 0
    try:
        for r in engine.generate(
            prompt=prompt,
            max_new_tokens=params.get("max_new_tokens", 4096),
            temperature=params.get("temperature", 0.01),
            top_p=params.get("top_p", 0.9),
            steering=steering,
            steering_threshold=threshold,
        ):
            msg = json.dumps({
                "type": "token",
                "token_id": r.token_id,
                "token_text": r.token_text,
                "position": r.position,
                "probe_score": r.probe_score,
                "was_steered": r.was_steered,
            })
            try:
                await ws.send(msg)
            except ConnectionClosed:
                engine.request_stop()
                return
            total += 1

        await ws.send(json.dumps({
            "type": "done",
            "total_tokens": total,
        }))
    except ConnectionClosed:
        engine.request_stop()


async def handler(ws, engine: GenerationEngine):
    """Handle a single WebSocket connection."""
    async for raw in ws:
        msg = handle_message(raw)
        action = msg.get("action")

        if action == "generate":
            await run_generation(ws, engine, msg)
        elif action == "stop":
            engine.request_stop()
        elif action == "update_steering":
            # Mid-generation steering updates are handled by the next generate call
            pass


async def start_server(host: str, port: int, engine: GenerationEngine):
    """Start the WebSocket server."""
    async def ws_handler(ws):
        await handler(ws, engine)

    server = await websockets.serve(ws_handler, host, port, ping_timeout=None)
    print(f"Inference worker listening on ws://{host}:{port}")
    await server.wait_closed()


def main():
    parser = argparse.ArgumentParser(description="Start inference worker")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--probe", type=str, default=None, help="Path to saved probe checkpoint")
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype (bfloat16, float32, etc.)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    dtype = resolve_torch_dtype(args.dtype)
    model, tokenizer = load_model_and_tokenizer(args.model, torch_dtype=dtype)

    probe = None
    if args.probe:
        print(f"Loading probe from {args.probe}...")
        probe = SequenceProbe.load(args.probe, model)

    engine = GenerationEngine(model=model, tokenizer=tokenizer, probe=probe)
    print("Model loaded. Starting server...")
    asyncio.run(start_server(args.host, args.port, engine))


if __name__ == "__main__":
    main()
