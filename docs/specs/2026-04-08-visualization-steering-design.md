# Degeneration Probe — Visualization & Model Steering Design

## Overview

An interactive system for real-time LLM degeneration detection and steering. Users observe token-by-token generation with live probe scores, adjust generation parameters, and enable active intervention when degeneration is detected.

## Architecture

Three-tier system:

```
Gradio UI (local) --HTTP/WS--> FastAPI Backend (local :8000) --WS over SSH tunnel--> Inference Worker (Clariden GPU :9000)
```

### 1. Gradio UI

Presentation layer only. Connects to FastAPI via HTTP (control) and WebSocket (token stream).

**Layout:**

```
+-----------------------------------------------------+
|  Connection Bar                                      |
|  [Worker Host: localhost] [Port: 9000] [Connect]     |
|  Status: * Connected  /  o Disconnected              |
+-----------------------------------------------------+
|  Prompt Input                                        |
|  [ Text area for free-text prompt                  ] |
|  [ Dataset dropdown: Alpaca / Hermes / AIME ]        |
|  [> Generate]  [# Stop]                              |
+------------------------+----------------------------+
|  Controls              |  Streaming Output           |
|                        |                             |
|  Temperature [===|==]  |  Tokens appear here, color- |
|  Top-p       [====|=]  |  coded green->yellow->red   |
|  Max tokens  [==|===]  |  by probe degeneration      |
|                        |  score. Steered tokens get   |
|  -- Steering --        |  an underline.              |
|  [x] Enable steering   |                             |
|  Strategy: [Temp Boost] |                            |
|  Threshold  [===|==]   |                             |
|  Boost temp [====|=]   |                             |
+------------------------+----------------------------+
|  Probe Score Bar                                     |
|  Live sparkline that grows with generation.          |
|  Threshold shown as dashed horizontal line.          |
|  Current score displayed numerically.                |
+-----------------------------------------------------+
```

**Visual encodings:**
- Token color: continuous green (0) -> yellow (0.5) -> red (1.0) mapped to probe score
- Steered tokens: underlined or bordered to distinguish intervention points
- Probe score bar: horizontal sparkline/bar chart growing with generation, threshold as dashed line
- Connection status: green dot (connected) / gray dot (disconnected)

### 2. FastAPI Backend

Local Python server. Manages sessions, relays token streams, stores generation history.

**Database (SQLite):**

`sessions` table:
- `id`, `created_at`, `worker_host`, `worker_port`, `status` (connected/disconnected)

`generations` table:
- `id`, `session_id`, `prompt`, `params_json`, `steering_json`, `output_text`, `tokens_json` (array of {token, probe_score, was_steered}), `status` (running/completed/stopped), `created_at`

**REST Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/sessions | Connect to inference worker (host, port) |
| GET | /api/sessions/current | Get current connection status |
| DELETE | /api/sessions/current | Disconnect from worker |
| GET | /api/generations | List past generations (paginated) |
| GET | /api/generations/{id} | Get full generation with per-token data |
| GET | /api/strategies | List available steering strategies |
| GET | /api/health | Backend health + worker connectivity check |

**WebSocket Endpoint:**

| Path | Purpose |
|------|---------|
| WS /api/generate | Start generation, stream tokens back. Client sends the generation request, then receives per-token messages. Client can send stop or update_steering mid-stream. |

### 3. Inference Worker

Long-running Python process that holds the model + probe in memory and exposes a WebSocket server on a fixed port (default 9000).

**Two deployment modes:**

- **Remote (Clariden):** Started via `salloc` or persistent SLURM job on a GPU node. Runs Apertus 8B. Backend connects via SSH tunnel: `ssh -L 9000:localhost:9000 clariden-gpu-node`
- **Local fallback:** Same worker code, started locally with a small model (e.g., Qwen/Qwen2.5-0.5B on CPU/MPS). Slower and less interesting outputs, but the full pipeline works — generation, probe scoring, steering. No tunnel needed, backend connects to localhost:9000 directly.

**Per-token generation flow:**

```
forward pass -> hook captures hidden state at probe layer ->
  probe scores hidden state -> if score > threshold: modify logits ->
  sample token -> stream {token, probe_score, was_steered} to backend
```

**Message Protocol:**

Backend -> Worker (start generation):
```json
{
  "action": "generate",
  "prompt": "Explain quantum computing...",
  "params": {
    "temperature": 0.01,
    "top_p": 0.9,
    "max_new_tokens": 4096
  },
  "steering": {
    "enabled": true,
    "strategy": "temperature_boost",
    "threshold": 0.8,
    "boost_temperature": 1.5
  }
}
```

Worker -> Backend (per token):
```json
{
  "type": "token",
  "token_id": 1234,
  "token_text": " quantum",
  "position": 42,
  "probe_score": 0.23,
  "was_steered": false
}
```

Worker -> Backend (generation complete):
```json
{
  "type": "done",
  "total_tokens": 512,
  "chunk_metrics": {}
}
```

Backend -> Worker (mid-generation control):
```json
{ "action": "stop" }
```
```json
{ "action": "update_steering", "steering": { "enabled": false } }
```

### 4. Steering Strategies

Extensible strategy interface:

```python
class SteeringStrategy(ABC):
    def should_intervene(self, probe_score: float, threshold: float) -> bool
    def intervene(self, logits: Tensor, context: SteeringContext) -> Tensor
```

**Initial implementation:** `TemperatureBoostStrategy` -- when probe_score exceeds threshold, scale logits by `1/boost_temperature` before sampling.

**Future strategies:** `RepetitionPenaltyStrategy` (penalize recent n-grams in logits), `HiddenStatePerturbation` (modify hidden state vector before remaining layers).

The UI exposes a dropdown to select the active strategy, and strategy-specific parameters appear as sliders below.

## User Stories

### 1. "I want to see degeneration happen"
User picks a prompt from Alpaca, sets temperature to 0.01, disables steering, hits Generate. Watches tokens stream in, sees them gradually turn red as the model enters a loop. The probe score bar spikes. Clear, visceral demonstration.

### 2. "I want to see steering fix it"
Same prompt, enables steering with temperature boost at threshold 0.8. Generates again. When tokens start going yellow/red, steering kicks in (tokens get underlined), and the model breaks out of the loop. Tokens return to green.

### 3. "I want to explore parameter sensitivity"
User adjusts the threshold slider to see how aggressive steering needs to be. Low threshold = intervenes early, output more diverse but less coherent. High threshold = only intervenes on severe loops.

### 4. "No cluster GPU, running locally"
User starts the worker locally with Qwen 0.5B. Connects the UI to localhost:9000. Same workflow as with Clariden — generation is slower and the model is less capable, but the full probe + steering loop works for development and demos.

## Tech Stack

- **Frontend:** Gradio
- **Backend:** FastAPI + SQLite
- **Inference Worker:** PyTorch + HuggingFace Transformers + existing degeneration_probe modules
- **Communication:** WebSocket (backend <-> worker via SSH tunnel), HTTP + WebSocket (Gradio <-> backend)
- **Model:** Swiss-AI/Apertus-8B-Instruct (Clariden), Qwen/Qwen2.5-0.5B for local dev/testing

## Development Notes

- **Local dev:** Run the inference worker locally with Qwen/Qwen2.5-0.5B (fits in CPU/MPS memory). Same protocol, same code paths, just a smaller model. Use base models at T=0.01 to reliably trigger degeneration for testing.
