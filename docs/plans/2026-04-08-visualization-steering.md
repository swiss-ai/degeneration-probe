# Visualization & Model Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-tier system (Gradio UI + FastAPI backend + inference worker) for real-time LLM degeneration detection and steering, with token-by-token streaming and probe-based intervention.

**Architecture:** FastAPI backend manages sessions and relays token streams between a Gradio UI and a remote/local inference worker. The worker runs token-by-token generation with a SequenceProbe hook, streaming per-token degeneration scores and applying configurable steering strategies when scores exceed a threshold.

**Tech Stack:** FastAPI, SQLite (via aiosqlite), websockets, Gradio, PyTorch, HuggingFace Transformers, existing `degeneration_probe` modules.

**Spec:** `docs/specs/2026-04-08-visualization-steering-design.md`

---

## File Structure

```
src/degeneration_probe/
├── server/
│   ├── __init__.py          # (empty)
│   ├── app.py               # FastAPI app factory, mounts routes
│   ├── database.py          # SQLite schema + CRUD operations
│   ├── routes_sessions.py   # POST/GET/DELETE /api/sessions
│   ├── routes_generations.py # GET /api/generations
│   ├── routes_health.py     # GET /api/health, GET /api/strategies
│   └── ws_generate.py       # WS /api/generate — relay between Gradio and worker
├── worker/
│   ├── __init__.py          # (empty)
│   ├── serve.py             # WebSocket server entry point (CLI)
│   ├── engine.py            # Token-by-token generation loop with probe + steering
│   └── steering.py          # SteeringStrategy ABC + TemperatureBoostStrategy
├── ui/
│   ├── __init__.py          # (empty)
│   └── app.py               # Gradio interface
tests/
├── test_steering.py         # Unit tests for steering strategies
├── test_engine.py           # Unit tests for generation engine
├── test_database.py         # Unit tests for database CRUD
├── test_routes.py           # Integration tests for FastAPI endpoints
└── test_worker_protocol.py  # Integration test for worker WebSocket protocol
```

---

### Task 1: Dependencies & Project Setup

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add new dependencies to pyproject.toml**

Add `fastapi`, `uvicorn`, `aiosqlite`, `websockets`, and `gradio` to the dependencies list in `pyproject.toml`:

```toml
dependencies = [
  "torch>=2.4.0",
  "transformers>=4.56.0",
  "accelerate>=0.20.0",
  "datasets>=2.0.0",
  "matplotlib>=3.8.0",
  "pyyaml>=6.0",
  "scikit-learn>=1.3.0",
  "wandb>=0.15.0",
  "openai>=1.0.0",
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "aiosqlite>=0.20.0",
  "websockets>=13.0",
  "gradio>=5.0.0",
]
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`
Expected: All new packages install successfully.

- [ ] **Step 3: Create package directories**

```bash
mkdir -p src/degeneration_probe/server
mkdir -p src/degeneration_probe/worker
mkdir -p src/degeneration_probe/ui
touch src/degeneration_probe/server/__init__.py
touch src/degeneration_probe/worker/__init__.py
touch src/degeneration_probe/ui/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/degeneration_probe/server/__init__.py src/degeneration_probe/worker/__init__.py src/degeneration_probe/ui/__init__.py
git commit -m "feat: add server/worker/ui packages and dependencies"
```

---

### Task 2: Steering Strategies

**Files:**
- Create: `src/degeneration_probe/worker/steering.py`
- Create: `tests/test_steering.py`

- [ ] **Step 1: Write failing tests for steering strategies**

Create `tests/test_steering.py`:

```python
"""Tests for steering strategies."""

import torch
import pytest

from degeneration_probe.worker.steering import (
    SteeringContext,
    SteeringStrategy,
    TemperatureBoostStrategy,
    get_strategy,
)


def test_temperature_boost_should_intervene_above_threshold():
    strategy = TemperatureBoostStrategy(boost_temperature=1.5)
    assert strategy.should_intervene(probe_score=0.9, threshold=0.8) is True


def test_temperature_boost_should_not_intervene_below_threshold():
    strategy = TemperatureBoostStrategy(boost_temperature=1.5)
    assert strategy.should_intervene(probe_score=0.5, threshold=0.8) is False


def test_temperature_boost_should_not_intervene_at_threshold():
    strategy = TemperatureBoostStrategy(boost_temperature=1.5)
    assert strategy.should_intervene(probe_score=0.8, threshold=0.8) is False


def test_temperature_boost_modifies_logits():
    strategy = TemperatureBoostStrategy(boost_temperature=2.0)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ctx = SteeringContext(recent_token_ids=[], position=10)
    result = strategy.intervene(logits, ctx)
    expected = logits / 2.0
    assert torch.allclose(result, expected)


def test_temperature_boost_preserves_shape():
    strategy = TemperatureBoostStrategy(boost_temperature=1.5)
    logits = torch.randn(50257)
    ctx = SteeringContext(recent_token_ids=[], position=0)
    result = strategy.intervene(logits, ctx)
    assert result.shape == logits.shape


def test_get_strategy_returns_temperature_boost():
    strategy = get_strategy("temperature_boost", boost_temperature=2.0)
    assert isinstance(strategy, TemperatureBoostStrategy)


def test_get_strategy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("unknown_strategy")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_steering.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degeneration_probe.worker.steering'`

- [ ] **Step 3: Implement steering strategies**

Create `src/degeneration_probe/worker/steering.py`:

```python
"""Steering strategies for probe-guided generation intervention."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class SteeringContext:
    """Context passed to steering strategies during intervention."""
    recent_token_ids: list[int]
    position: int


class SteeringStrategy(ABC):
    """Base class for steering strategies."""

    @abstractmethod
    def should_intervene(self, probe_score: float, threshold: float) -> bool:
        ...

    @abstractmethod
    def intervene(self, logits: torch.Tensor, context: SteeringContext) -> torch.Tensor:
        ...


class TemperatureBoostStrategy(SteeringStrategy):
    """When probe score exceeds threshold, divide logits by boost_temperature."""

    def __init__(self, boost_temperature: float = 1.5):
        self.boost_temperature = boost_temperature

    def should_intervene(self, probe_score: float, threshold: float) -> bool:
        return probe_score > threshold

    def intervene(self, logits: torch.Tensor, context: SteeringContext) -> torch.Tensor:
        return logits / self.boost_temperature


STRATEGY_REGISTRY: dict[str, type[SteeringStrategy]] = {
    "temperature_boost": TemperatureBoostStrategy,
}


def get_strategy(name: str, **kwargs) -> SteeringStrategy:
    """Instantiate a steering strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name](**kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_steering.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/degeneration_probe/worker/steering.py tests/test_steering.py
git commit -m "feat: add SteeringStrategy ABC and TemperatureBoostStrategy"
```

---

### Task 3: Token-by-Token Generation Engine

**Files:**
- Create: `src/degeneration_probe/worker/engine.py`
- Create: `tests/test_engine.py`

This is the core component — it replaces `generate_for_prompt()` with a token-by-token loop that runs the probe at each step and applies steering.

- [ ] **Step 1: Write failing tests for the generation engine**

Create `tests/test_engine.py`:

```python
"""Tests for the token-by-token generation engine."""

import pytest

from degeneration_probe.worker.engine import GenerationEngine, TokenResult


class FakeModel:
    """Minimal fake model for testing the engine without GPU."""

    class FakeConfig:
        hidden_size = 64

    def __init__(self):
        self.config = self.FakeConfig()
        self.device = "cpu"

    def __call__(self, **kwargs):
        import torch

        batch = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        # Return a fake output with logits and hidden states
        logits = torch.randn(batch, seq_len, 100)
        return type("Output", (), {"logits": logits})()

    def parameters(self):
        return iter([])


class FakeTokenizer:
    """Minimal fake tokenizer."""

    eos_token_id = 99
    pad_token = "<pad>"

    def __call__(self, text, return_tensors=None, **kwargs):
        import torch
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(self, messages, **kwargs):
        import torch
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        return type("Enc", (), {"to": lambda self, d: self, "input_ids": ids, "attention_mask": mask})()

    def decode(self, ids, **kwargs):
        return " ".join(f"tok{i}" for i in ids)


def test_token_result_fields():
    r = TokenResult(
        token_id=42,
        token_text="hello",
        position=0,
        probe_score=0.5,
        was_steered=False,
    )
    assert r.token_id == 42
    assert r.probe_score == 0.5


def test_engine_init():
    engine = GenerationEngine(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        probe=None,
    )
    assert engine.model is not None
    assert engine.tokenizer is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degeneration_probe.worker.engine'`

- [ ] **Step 3: Implement the generation engine**

Create `src/degeneration_probe/worker/engine.py`:

```python
"""Token-by-token generation engine with probe scoring and steering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from degeneration_probe.probe import SequenceProbe
from degeneration_probe.worker.steering import SteeringContext, SteeringStrategy


@dataclass
class TokenResult:
    """Result for a single generated token."""
    token_id: int
    token_text: str
    position: int
    probe_score: float
    was_steered: bool


class GenerationEngine:
    """Runs token-by-token generation with probe scoring and optional steering.

    The engine holds the model, tokenizer, and optionally a trained probe.
    For each generation request it:
    1. Encodes the prompt
    2. In a loop: runs one forward pass, hooks the probe layer, scores,
       optionally steers logits, samples the next token, yields the result.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        probe: Optional[SequenceProbe] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self._stop_requested = False

    def request_stop(self):
        """Signal the generation loop to stop after the current token."""
        self._stop_requested = True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.01,
        top_p: float = 0.9,
        steering: Optional[SteeringStrategy] = None,
        steering_threshold: float = 0.8,
    ) -> list[TokenResult]:
        """Synchronous token-by-token generation. Returns list of TokenResult.

        Used by the WebSocket worker serve loop, which sends each result as it
        is produced.
        """
        self._stop_requested = False
        device = getattr(self.model, "device", torch.device("cpu"))

        # Encode prompt
        messages = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        attention_mask = (
            encoded["attention_mask"]
            if isinstance(encoded, dict)
            else getattr(encoded, "attention_mask", torch.ones_like(input_ids))
        )

        generated_ids: list[int] = []
        results: list[TokenResult] = []

        for pos in range(max_new_tokens):
            if self._stop_requested:
                break

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            # Get logits for the last position
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            # Probe scoring
            probe_score = 0.0
            if self.probe is not None and self.probe._hooked is not None:
                hidden = self.probe._hooked[:, -1, :]  # [1, H]
                if hidden.dtype != self.probe.linear.weight.dtype:
                    hidden = hidden.to(self.probe.linear.weight.dtype)
                logit = self.probe.linear(hidden)  # [1, 1]
                probe_score = torch.sigmoid(logit).item()

            # Steering
            was_steered = False
            if steering is not None and steering.should_intervene(probe_score, steering_threshold):
                ctx = SteeringContext(
                    recent_token_ids=generated_ids[-50:],
                    position=pos,
                )
                logits = steering.intervene(logits.squeeze(0), ctx).unsqueeze(0)
                was_steered = True

            # Sample
            if temperature < 0.02:
                next_token_id = logits.argmax(dim=-1).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative = sorted_probs.cumsum(dim=-1)
                    mask = cumulative - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    idx = torch.multinomial(sorted_probs, 1)
                    next_token_id = sorted_indices.gather(-1, idx).item()
                else:
                    next_token_id = torch.multinomial(probs, 1).item()

            # EOS check
            if next_token_id == self.tokenizer.eos_token_id:
                break

            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            generated_ids.append(next_token_id)

            result = TokenResult(
                token_id=next_token_id,
                token_text=token_text,
                position=pos,
                probe_score=probe_score,
                was_steered=was_steered,
            )
            results.append(result)

            # Extend input for next iteration (KV cache is not used here for simplicity)
            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_engine.py -v`
Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/degeneration_probe/worker/engine.py tests/test_engine.py
git commit -m "feat: add token-by-token GenerationEngine with probe scoring"
```

---

### Task 4: Inference Worker WebSocket Server

**Files:**
- Create: `src/degeneration_probe/worker/serve.py`
- Create: `tests/test_worker_protocol.py`

The worker is a standalone process that loads the model+probe and serves generation requests over WebSocket.

- [ ] **Step 1: Write failing test for the worker protocol**

Create `tests/test_worker_protocol.py`:

```python
"""Integration test for the inference worker WebSocket protocol."""

import asyncio
import json
import pytest

from degeneration_probe.worker.serve import handle_message


def test_handle_message_parses_generate_action():
    msg = json.dumps({
        "action": "generate",
        "prompt": "Hello",
        "params": {"temperature": 0.5, "top_p": 0.9, "max_new_tokens": 10},
        "steering": {"enabled": False},
    })
    parsed = json.loads(msg)
    assert parsed["action"] == "generate"
    assert parsed["prompt"] == "Hello"


def test_handle_message_parses_stop_action():
    msg = json.dumps({"action": "stop"})
    parsed = json.loads(msg)
    assert parsed["action"] == "stop"


def test_handle_message_parses_update_steering():
    msg = json.dumps({
        "action": "update_steering",
        "steering": {"enabled": True, "strategy": "temperature_boost", "threshold": 0.7},
    })
    parsed = json.loads(msg)
    assert parsed["action"] == "update_steering"
    assert parsed["steering"]["strategy"] == "temperature_boost"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_worker_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degeneration_probe.worker.serve'`

- [ ] **Step 3: Implement the worker WebSocket server**

Create `src/degeneration_probe/worker/serve.py`:

```python
"""Inference worker: WebSocket server that runs generation on GPU."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch
import websockets

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

    results = engine.generate(
        prompt=prompt,
        max_new_tokens=params.get("max_new_tokens", 4096),
        temperature=params.get("temperature", 0.01),
        top_p=params.get("top_p", 0.9),
        steering=steering,
        steering_threshold=threshold,
    )

    for r in results:
        msg = json.dumps({
            "type": "token",
            "token_id": r.token_id,
            "token_text": r.token_text,
            "position": r.position,
            "probe_score": r.probe_score,
            "was_steered": r.was_steered,
        })
        await ws.send(msg)

    await ws.send(json.dumps({
        "type": "done",
        "total_tokens": len(results),
    }))


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

    server = await websockets.serve(ws_handler, host, port)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_worker_protocol.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/degeneration_probe/worker/serve.py tests/test_worker_protocol.py
git commit -m "feat: add inference worker WebSocket server"
```

---

### Task 5: Database Layer

**Files:**
- Create: `src/degeneration_probe/server/database.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: Write failing tests for database operations**

Create `tests/test_database.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_database.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degeneration_probe.server.database'`

- [ ] **Step 3: Implement the database layer**

Create `src/degeneration_probe/server/database.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_database.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/degeneration_probe/server/database.py tests/test_database.py
git commit -m "feat: add SQLite database layer for sessions and generations"
```

---

### Task 6: FastAPI Routes

**Files:**
- Create: `src/degeneration_probe/server/app.py`
- Create: `src/degeneration_probe/server/routes_sessions.py`
- Create: `src/degeneration_probe/server/routes_generations.py`
- Create: `src/degeneration_probe/server/routes_health.py`
- Create: `src/degeneration_probe/server/ws_generate.py`
- Create: `tests/test_routes.py`

- [ ] **Step 1: Write failing tests for the REST endpoints**

Create `tests/test_routes.py`:

```python
"""Integration tests for FastAPI routes."""

import pytest
from fastapi.testclient import TestClient

from degeneration_probe.server.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(db_path=tmp_path / "test.db")
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_strategies(client):
    resp = client.get("/api/strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert "temperature_boost" in data


def test_create_session(client):
    resp = client.post("/api/sessions", json={"worker_host": "localhost", "worker_port": 9000})
    assert resp.status_code == 200
    data = resp.json()
    assert data["worker_host"] == "localhost"
    assert data["status"] == "connected"


def test_get_current_session(client):
    client.post("/api/sessions", json={"worker_host": "localhost", "worker_port": 9000})
    resp = client.get("/api/sessions/current")
    assert resp.status_code == 200
    assert resp.json()["worker_host"] == "localhost"


def test_get_current_session_when_none(client):
    resp = client.get("/api/sessions/current")
    assert resp.status_code == 404


def test_delete_session(client):
    client.post("/api/sessions", json={"worker_host": "localhost", "worker_port": 9000})
    resp = client.delete("/api/sessions/current")
    assert resp.status_code == 200
    resp2 = client.get("/api/sessions/current")
    assert resp2.status_code == 404


def test_list_generations_empty(client):
    resp = client.get("/api/generations")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_generation_not_found(client):
    resp = client.get("/api/generations/999")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degeneration_probe.server.app'`

- [ ] **Step 3: Implement route modules**

Create `src/degeneration_probe/server/routes_health.py`:

```python
"""Health and metadata endpoints."""

from fastapi import APIRouter

from degeneration_probe.worker.steering import STRATEGY_REGISTRY

router = APIRouter()


@router.get("/api/health")
async def health():
    return {"status": "ok"}


@router.get("/api/strategies")
async def strategies():
    return {
        name: {"description": cls.__doc__ or name}
        for name, cls in STRATEGY_REGISTRY.items()
    }
```

Create `src/degeneration_probe/server/routes_sessions.py`:

```python
"""Session management endpoints."""

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
```

Create `src/degeneration_probe/server/routes_generations.py`:

```python
"""Generation history endpoints."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/api/generations")
async def list_generations(request: Request, limit: int = 20, offset: int = 0):
    db = request.app.state.db
    return await db.list_generations(limit=limit, offset=offset)


@router.get("/api/generations/{gen_id}")
async def get_generation(gen_id: int, request: Request):
    db = request.app.state.db
    gen = await db.get_generation(gen_id)
    if gen is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return gen
```

Create `src/degeneration_probe/server/ws_generate.py`:

```python
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
```

Create `src/degeneration_probe/server/app.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_routes.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/degeneration_probe/server/app.py src/degeneration_probe/server/routes_health.py src/degeneration_probe/server/routes_sessions.py src/degeneration_probe/server/routes_generations.py src/degeneration_probe/server/ws_generate.py tests/test_routes.py
git commit -m "feat: add FastAPI backend with REST and WebSocket endpoints"
```

---

### Task 7: Gradio UI

**Files:**
- Create: `src/degeneration_probe/ui/app.py`

This task builds the Gradio interface. It connects to the FastAPI backend via HTTP for control operations and WebSocket for streaming generation.

- [ ] **Step 1: Implement the Gradio UI**

Create `src/degeneration_probe/ui/app.py`:

```python
"""Gradio interface for the degeneration probe visualization."""

from __future__ import annotations

import json

import gradio as gr
import requests
import websocket  # from websocket-client, bundled with gradio

API_BASE = "http://localhost:8000"


def connect_worker(host: str, port: int):
    """Connect to an inference worker."""
    try:
        resp = requests.post(
            f"{API_BASE}/api/sessions",
            json={"worker_host": host, "worker_port": port},
            timeout=5,
        )
        if resp.ok:
            return f"Connected to {host}:{port}"
        return f"Failed: {resp.text}"
    except Exception as e:
        return f"Connection error: {e}"


def disconnect_worker():
    """Disconnect from the current worker."""
    try:
        resp = requests.delete(f"{API_BASE}/api/sessions/current", timeout=5)
        return "Disconnected" if resp.ok else f"Failed: {resp.text}"
    except Exception as e:
        return f"Error: {e}"


def get_status():
    """Check current connection status."""
    try:
        resp = requests.get(f"{API_BASE}/api/sessions/current", timeout=2)
        if resp.ok:
            s = resp.json()
            return f"Connected to {s['worker_host']}:{s['worker_port']}"
        return "Disconnected"
    except Exception:
        return "Backend unavailable"


def generate_stream(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    steering_enabled: bool,
    strategy: str,
    threshold: float,
    boost_temp: float,
):
    """Stream generation from the worker via the backend WebSocket."""
    ws_url = f"ws://localhost:8000/api/generate"

    request = {
        "action": "generate",
        "prompt": prompt,
        "params": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": int(max_tokens),
        },
        "steering": {
            "enabled": steering_enabled,
            "strategy": strategy,
            "threshold": threshold,
            "boost_temperature": boost_temp,
        },
    }

    tokens_html = ""
    scores = []

    try:
        ws = websocket.create_connection(ws_url)
        ws.send(json.dumps(request))

        while True:
            raw = ws.recv()
            data = json.loads(raw)

            if data["type"] == "error":
                yield tokens_html + f"<br><b>Error:</b> {data['message']}", ""
                break

            if data["type"] == "done":
                break

            if data["type"] == "token":
                score = data["probe_score"]
                scores.append(score)
                steered = data["was_steered"]

                # Color: green (0) -> yellow (0.5) -> red (1.0)
                if score < 0.5:
                    r = int(255 * score * 2)
                    g = 200
                else:
                    r = 255
                    g = int(200 * (1 - (score - 0.5) * 2))
                color = f"rgb({r},{g},0)"

                style = f"color:{color};"
                if steered:
                    style += "text-decoration:underline;"

                token_text = data["token_text"].replace("<", "&lt;").replace(">", "&gt;")
                tokens_html += f'<span style="{style}" title="score={score:.3f}">{token_text}</span>'

                # Build sparkline as simple bar chart
                bar_html = _build_score_bar(scores, threshold)

                yield tokens_html, bar_html

        ws.close()

    except Exception as e:
        yield tokens_html + f"<br><b>Connection error:</b> {e}", ""


def _build_score_bar(scores: list[float], threshold: float) -> str:
    """Build an HTML sparkline bar chart of probe scores."""
    if not scores:
        return ""
    bar_width = max(2, min(6, 600 // len(scores)))
    bars = []
    for s in scores:
        height = max(2, int(s * 60))
        if s < 0.5:
            r = int(255 * s * 2)
            g = 200
        else:
            r = 255
            g = int(200 * (1 - (s - 0.5) * 2))
        color = f"rgb({r},{g},0)"
        bars.append(
            f'<div style="display:inline-block;width:{bar_width}px;height:{height}px;'
            f'background:{color};vertical-align:bottom;"></div>'
        )
    threshold_y = int(threshold * 60)
    return (
        f'<div style="position:relative;height:65px;overflow-x:auto;white-space:nowrap;'
        f'border-bottom:1px solid #ccc;padding-top:5px;">'
        f'<div style="position:absolute;top:{65-threshold_y}px;left:0;right:0;'
        f'border-top:2px dashed rgba(255,0,0,0.5);"></div>'
        f'{"".join(bars)}'
        f'</div>'
        f'<div style="font-size:12px;color:#666;">Current: {scores[-1]:.3f} | Threshold: {threshold:.2f}</div>'
    )


def build_ui():
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(title="Degeneration Probe", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Degeneration Probe — Live Steering")

        # Connection bar
        with gr.Row():
            host_input = gr.Textbox(value="localhost", label="Worker Host", scale=2)
            port_input = gr.Number(value=9000, label="Port", precision=0, scale=1)
            connect_btn = gr.Button("Connect", scale=1)
            disconnect_btn = gr.Button("Disconnect", scale=1)
            status_display = gr.Textbox(label="Status", interactive=False, scale=2)

        connect_btn.click(
            connect_worker, inputs=[host_input, port_input], outputs=[status_display]
        )
        disconnect_btn.click(disconnect_worker, outputs=[status_display])
        demo.load(get_status, outputs=[status_display])

        # Prompt input
        with gr.Row():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter a prompt or select from dataset samples...",
                lines=3,
                scale=4,
            )
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            stop_btn = gr.Button("Stop")

        # Controls + Output
        with gr.Row():
            # Left: controls
            with gr.Column(scale=1):
                gr.Markdown("### Generation Parameters")
                temperature = gr.Slider(0.0, 2.0, value=0.01, step=0.01, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                max_tokens = gr.Slider(64, 4096, value=512, step=64, label="Max Tokens")

                gr.Markdown("### Steering")
                steering_enabled = gr.Checkbox(label="Enable Steering", value=False)
                strategy = gr.Dropdown(
                    choices=["temperature_boost"],
                    value="temperature_boost",
                    label="Strategy",
                )
                threshold = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Threshold")
                boost_temp = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="Boost Temperature")

            # Right: streaming output
            with gr.Column(scale=3):
                output_html = gr.HTML(label="Generated Text")
                score_bar = gr.HTML(label="Probe Score")

        generate_btn.click(
            generate_stream,
            inputs=[
                prompt_input, temperature, top_p, max_tokens,
                steering_enabled, strategy, threshold, boost_temp,
            ],
            outputs=[output_html, score_bar],
        )

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports without errors**

Run: `uv run python -c "from degeneration_probe.ui.app import build_ui; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/degeneration_probe/ui/app.py
git commit -m "feat: add Gradio UI with streaming token display and steering controls"
```

---

### Task 8: CLI Entry Points

**Files:**
- Modify: `src/degeneration_probe/__main__.py` (add `serve` and `ui` subcommands)

- [ ] **Step 1: Add `serve` and `ui` commands to the CLI**

Add these two new command functions and register them in the argument parser. Add them _after_ the existing `cmd_evaluate` function (after line 252 in `__main__.py`) and register them alongside the existing subcommands.

Add the command functions after `cmd_evaluate`:

```python
def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI backend server."""
    import uvicorn
    from degeneration_probe.server.app import create_app

    app = create_app(db_path=args.db_path)
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_worker(args: argparse.Namespace) -> None:
    """Start the inference worker."""
    from degeneration_probe.worker.serve import main as worker_main
    import sys
    # Re-inject args so worker's argparse sees them
    sys.argv = ["worker", "--model", args.model, "--host", args.host, "--port", str(args.port)]
    if args.probe:
        sys.argv.extend(["--probe", args.probe])
    if args.dtype:
        sys.argv.extend(["--dtype", args.dtype])
    worker_main()


def cmd_ui(args: argparse.Namespace) -> None:
    """Start the Gradio UI."""
    from degeneration_probe.ui.app import build_ui
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port)
```

Register the subcommands in `main()` alongside the existing ones:

```python
    # Add after the evaluate subparser
    sp_serve = sub.add_parser("serve", help="Start the FastAPI backend server")
    sp_serve.add_argument("--host", default="0.0.0.0")
    sp_serve.add_argument("--port", type=int, default=8000)
    sp_serve.add_argument("--db-path", default="data/degeneration_probe.db")
    sp_serve.set_defaults(func=cmd_serve)

    sp_worker = sub.add_parser("worker", help="Start the inference worker")
    sp_worker.add_argument("--model", required=True, help="HuggingFace model name")
    sp_worker.add_argument("--probe", default=None, help="Path to saved probe checkpoint")
    sp_worker.add_argument("--dtype", default=None, help="Model dtype")
    sp_worker.add_argument("--host", default="0.0.0.0")
    sp_worker.add_argument("--port", type=int, default=9000)
    sp_worker.set_defaults(func=cmd_worker)

    sp_ui = sub.add_parser("ui", help="Start the Gradio UI")
    sp_ui.add_argument("--host", default="0.0.0.0")
    sp_ui.add_argument("--port", type=int, default=7860)
    sp_ui.set_defaults(func=cmd_ui)
```

- [ ] **Step 2: Verify the CLI parses correctly**

Run: `uv run python -m degeneration_probe serve --help`
Expected: Shows help for the serve command with `--host`, `--port`, `--db-path` options.

Run: `uv run python -m degeneration_probe worker --help`
Expected: Shows help for the worker command with `--model`, `--probe`, `--dtype`, `--host`, `--port` options.

Run: `uv run python -m degeneration_probe ui --help`
Expected: Shows help for the ui command with `--host`, `--port` options.

- [ ] **Step 3: Commit**

```bash
git add src/degeneration_probe/__main__.py
git commit -m "feat: add serve, worker, and ui CLI subcommands"
```

---

### Task 9: End-to-End Smoke Test

This task verifies all three tiers work together. No new files — just a manual verification sequence.

- [ ] **Step 1: Run all unit tests**

Run: `uv run pytest tests/ -v --ignore=tests/test_routes.py`
Expected: All tests in `test_steering.py`, `test_engine.py`, `test_database.py`, `test_worker_protocol.py` PASS.

Run: `uv run pytest tests/test_routes.py -v`
Expected: All route tests PASS.

- [ ] **Step 2: Start the backend in a terminal**

Run: `uv run python -m degeneration_probe serve --port 8000`
Expected: `Uvicorn running on http://0.0.0.0:8000`

Verify in a second terminal:
Run: `curl http://localhost:8000/api/health`
Expected: `{"status":"ok"}`

Run: `curl http://localhost:8000/api/strategies`
Expected: JSON with `temperature_boost` entry.

- [ ] **Step 3: Verify the UI starts**

In a third terminal:
Run: `uv run python -m degeneration_probe ui --port 7860`
Expected: `Running on local URL: http://0.0.0.0:7860`

Open the URL in a browser and verify the layout matches the spec: connection bar, prompt input, parameter sliders, steering controls, output area, and probe score bar.

- [ ] **Step 4: Commit any fixes**

If any adjustments were needed during smoke testing, commit them:

```bash
git add -u
git commit -m "fix: smoke test fixes for end-to-end integration"
```
