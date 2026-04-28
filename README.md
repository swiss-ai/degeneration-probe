# degeneration-probe

A research pipeline for collecting labelled LLM generation data and training a **degeneration probe** — a lightweight classifier that detects repetitive model outputs from internal hidden states, without needing to re-run the full metrics pipeline at inference time.

The repo has two tracks:

- **Data pipeline** (CLI): generate labelled completions → train a probe on hidden states → evaluate on held-out data.
- **Interactive UI** (backend + worker + Gradio UI): stream tokens from a live model while the probe scores each token, with optional steering to break out of degenerate loops.

Both tracks share the same codebase and CLI (`python -m degeneration_probe …`).

## Contents

- [Setup](#setup)
- [Building a probe (data pipeline)](#building-a-probe-data-pipeline)
  - [Step 1: Generate labelled data](#step-1-generate-labelled-data)
  - [Step 2: Train a probe](#step-2-train-a-probe)
  - [Step 3: Evaluate on new data](#step-3-evaluate-on-new-data)
  - [Creating your own configs](#creating-your-own-configs)
  - [Training on multiple datasets](#training-on-multiple-datasets)
  - [Dataset caching](#dataset-caching)
- [Interactive UI](#interactive-ui)
  - [Quick start (local)](#quick-start-local)
  - [Using a trained probe](#using-a-trained-probe)
  - [Steering](#steering)
  - [Running on CSCS Clariden](#running-on-cscs-clariden)
- [Reference](#reference)
  - [CLI commands](#cli-commands)
  - [API endpoints](#api-endpoints)
  - [Output format](#output-format)
  - [Metrics](#metrics)
  - [How probe training works](#how-probe-training-works)

---

## Setup

**Prerequisites**

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A HuggingFace account with an [access token](https://huggingface.co/settings/tokens) — needed to download models and datasets
- GPU recommended. The small Qwen-0.5B model runs on CPU/MPS but is slow.

**Install**

```bash
git clone <repo-url>
cd degeneration-probe

uv sync

# Save your HuggingFace token to .env (read by both local runs and cluster jobs).
echo 'HF_TOKEN=hf_YOUR_TOKEN_HERE' >> .env
```

---

## Building a probe (data pipeline)

### Step 1: Generate labelled data

Downloads prompts from a HuggingFace dataset, generates LLM completions, computes repetition metrics per completion, and labels each one as `degenerating: true/false`.

```bash
uv run python -m degeneration_probe generate --config configs/generate/alpaca_qwen05b.yaml
```

The config at `configs/generate/alpaca_qwen05b.yaml` generates 10 completions from **Qwen-0.5B** on **Alpaca** prompts. Open it to see all available options.

When the run finishes, it prints the output directory:

```
Run complete. Results saved to: outputs/generations/20260403_115426
```

Inside that directory:

- `data/generations.jsonl` — the labelled data (what you feed to training)
- `data/prompts.jsonl` — the raw prompts that were fetched
- `plots/` — histograms of TTR and repetition metrics
- `config.json` — exact config snapshot for reproducibility

### Step 2: Train a probe

Two ready-to-go training configs live in `configs/train/`:

- `qwen05b_hf.yaml` — Qwen-0.5B base, runs locally (CPU/MPS) or on cluster
- `apertus8b_hf.yaml` — Apertus-8B-Instruct, cluster-only (uses ~16 GB GPU)

Both pull data from the gated HF dataset `luca-sartori/degeneration-probe-instruct`. To train on your own `generations.jsonl` instead, copy one of them and replace the `hf_dataset:` block with `train_data: [path/to/generations.jsonl]`.

```bash
uv run python -m degeneration_probe train --config configs/train/qwen05b_hf.yaml
```

Output directory (e.g. `outputs/probes/20260428_123456`):

- `checkpoint/` — `probe_head.bin`, `adapter_model.safetensors`, `probe_config.json`, `degeneration_meta.json`
- `config.json` — full training config snapshot

Key config options:

- `probe.layer` — which transformer layer to probe (12 = middle of Qwen-0.5B; 16 = middle of Apertus-8B)
- `probe.lora.{rank,alpha,dropout}` — LoRA adapter config
- `label.{window_size,primary_n}` — sliding window (256 tokens) and n-gram size (1) for the per-token 1 - TTR target
- `learning_rate.{head,lora}` — separate LRs for the value head vs LoRA params
- `num_epochs`, `batch_size`, `max_length` — standard training hyperparameters
- `model.name` — required when training from `hf_dataset`; auto-resolved from JSONL records otherwise.

### Step 3: Evaluate on new data

Evaluate a trained probe on a different dataset:

```bash
uv run python -m degeneration_probe evaluate \
  --checkpoint outputs/probes/20260403_115847/checkpoint \
  --eval_data outputs/generations/20260403_120000/data/generations.jsonl
```

The model is loaded automatically from the saved training config. Results (metrics JSON, ROC curve, threshold plot) go to the checkpoint's `eval/` subdirectory by default, or to a custom location with `--output_dir`.

Additional options:

- `--batch_size N` (default: 4)
- `--max_length N` (default: 2048)
- `--model_name NAME` — override the model (only needed if the saved config is missing)
- `--output_dir PATH` — save to a custom directory instead of `checkpoint/eval/`

### Creating your own configs

**Generation config.** Copy an existing one:

```bash
cp configs/generate/alpaca_qwen05b.yaml configs/generate/my_experiment.yaml
```

Then edit the model and dataset sections:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct     # any HuggingFace causal LM
  dtype: bfloat16                     # bfloat16 (GPU), float32 (CPU/MPS), or auto
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9

dataset:
  name: openai/gsm8k                  # any HuggingFace dataset
  subset: main                        # dataset subset (null if none)
  split: test
  prompt_field: question              # which field holds the prompt (null = auto-detect)
  max_samples: 100                    # how many prompts to download
  max_prompts: 50                     # how many to actually generate for

analysis:
  chunk_size: 256                     # tokens per chunk for metrics
  n_values: [1, 3]                    # n-gram sizes for TTR
```

Then `uv run python -m degeneration_probe generate --config configs/generate/my_experiment.yaml`.

**Training config.** Same pattern:

```bash
cp configs/train/qwen05b_hf.yaml configs/train/my_training.yaml
```

Then edit `model.name`, `probe.layer`, and either `hf_dataset` or `train_data` (see the comments inside the file).

For local-only configs you don't want committed, name the file `*.local.yaml` — gitignored automatically.

### Training on multiple datasets

To make a probe more robust, train it on data from multiple datasets (e.g. Alpaca + AIME):

1. **Generate data separately for each dataset** — each run uses one dataset and one model. All runs must use the **same model** (e.g. `Qwen/Qwen2.5-0.5B-Instruct`); training validates this and will refuse to proceed if it's violated.

    ```bash
    uv run python -m degeneration_probe generate --config configs/generate/alpaca_qwen05b.yaml
    uv run python -m degeneration_probe generate --config configs/generate/aime_qwen05b.yaml
    ```

2. **List all JSONLs in `train_data`:**

    ```yaml
    train_data:
      - outputs/generations/20260403_100000/data/generations.jsonl   # Alpaca
      - outputs/generations/20260403_100500/data/generations.jsonl   # AIME
    ```

    The training script concatenates all records. `eval_fraction` is applied to the combined dataset.

3. **Train:** `uv run python -m degeneration_probe train --config …`.

4. **Evaluate per dataset** by passing one JSONL at a time to `evaluate`, with `--output_dir` so results don't overwrite each other:

    ```bash
    uv run python -m degeneration_probe evaluate \
      --checkpoint outputs/probes/20260403_110000/checkpoint \
      --eval_data outputs/generations/20260403_100000/data/generations.jsonl \
      --output_dir outputs/probes/20260403_110000/eval_alpaca
    ```

### Dataset caching

Prompt samples are cached in `data/cache/` (gitignored) so subsequent runs with the same config skip the HuggingFace download. The cache filename encodes all parameters that affect the sample — changing any of them produces a new cache file automatically.

Override the cache location (e.g. for a shared cluster filesystem):

```bash
export PROBE_DATA_CACHE=/path/to/shared/cache
```

---

## Interactive UI

The UI streams tokens from a live model and colours each one by its probe score. It has three components that run as separate processes:

| Component | Command                                     | Default port | Purpose                                  |
|-----------|---------------------------------------------|--------------|------------------------------------------|
| Backend   | `python -m degeneration_probe serve`        | 8000         | FastAPI server — REST + WebSocket relay  |
| Worker    | `python -m degeneration_probe worker --model …` | 9000     | Loads the model + probe, generates tokens |
| UI        | `python -m degeneration_probe ui`           | 7860         | Gradio web interface                     |

The UI connects to the backend on `localhost:8000` and expects the worker reachable at `localhost:9000` (either a local worker, or an SSH tunnel to a remote one).

### Quick start (local)

Open three terminals:

```bash
# Tab 1: backend
uv run python -m degeneration_probe serve

# Tab 2: worker (small CPU/MPS-friendly model)
uv run python -m degeneration_probe worker --model Qwen/Qwen2.5-0.5B-Instruct --dtype float32

# Tab 3: UI
uv run python -m degeneration_probe ui
```

Open **http://localhost:7860** in your browser, type a prompt, click **Generate**. The UI auto-connects to the local worker on startup — no manual connect step.

### Using a trained probe

Pass a checkpoint from the data pipeline to the worker:

```bash
uv run python -m degeneration_probe worker \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --probe outputs/probes/<timestamp>/checkpoint \
  --dtype float32
```

Tokens are coloured by probe score in the UI (green = safe, amber = borderline, red = degenerate).

### Steering

Enable steering in the UI sidebar to let the probe intervene during generation. When the probe score exceeds the threshold, the selected strategy modifies the model's output distribution to break out of repetitive loops.

Available strategies:

- **Temperature Boost** — divides logits by a higher temperature when degeneration is detected, increasing output diversity.

### Running on CSCS Clariden

For running Apertus-8B (or larger) on a Clariden GPU node, the repo ships a one-shot launcher that brings up the worker job, the SSH tunnel, the local backend, and the local UI:

```bash
./cluster/start.sh
```

Defaults target the `debug` partition with Apertus-8B. Override via env vars:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct PARTITION=normal TIME=04:00:00 ./cluster/start.sh
```

Available overrides: `MODEL`, `PARTITION`, `TIME`, `WORKER_PORT`, `BACKEND_PORT`, `UI_PORT`, `SSH_HOST`, `QUEUE_WAIT_SECS`.

The script prints the UI URL, backend URL, worker URL, SLURM job ID, and node name when the stack is up. Runtime state (PIDs, job ID, logs) lives in `.run/`.

**Tear down:**

```bash
./cluster/stop.sh
```

This kills the local processes, closes the tunnel, and cancels the SLURM job.

**Prerequisites**

- SSH access to Clariden configured (see the [swiss-ai setup guide](https://github.com/swiss-ai/documentation/blob/main/pages/setup_clariden.md)).
- The repo cloned on Clariden at `~/degeneration-probe` (`start.sh` hard-resets it to `origin/serving` on every launch).
- A container environment `my_env` set up in `~/.edf/`.
- Your `cscs-key` loaded in the SSH agent: `ssh-add -t 1d ~/.ssh/cscs-key`.

**Manual launch (fallback)**

If you want to launch the pieces by hand — e.g. to debug the worker in isolation — submit the worker job directly and wire up the tunnel yourself:

```bash
# On Clariden
ssh clariden
cd ~/degeneration-probe
sbatch cluster/clariden_worker.sh                       # defaults: Apertus-8B, no probe
PROBE=outputs/probes/20260408/checkpoint sbatch cluster/clariden_worker.sh
MODEL=Qwen/Qwen2.5-7B-Instruct sbatch cluster/clariden_worker.sh

# Find the node and open a tunnel (on your laptop)
cat /iopsstor/scratch/cscs/$USER/logs/worker_<jobid>.out   # look for the tunnel hint line
ssh -L 9000:<node>:9000 clariden

# Start backend + UI locally
uv run python -m degeneration_probe serve &
uv run python -m degeneration_probe ui &
```

Clean up with `scancel <jobid>` on Clariden.

---

## Reference

### CLI commands

All commands share the same entry point: `python -m degeneration_probe {generate,train,evaluate,serve,worker,ui}`.

| Command  | Purpose                                                              |
|----------|----------------------------------------------------------------------|
| `generate` | Download prompts, generate completions, compute metrics, label data |
| `train`    | Train a probe head on hidden states from labelled data              |
| `evaluate` | Evaluate a trained probe on a new JSONL file                        |
| `serve`    | Start the FastAPI backend (default `0.0.0.0:8000`)                  |
| `worker`   | Start the inference worker (default `0.0.0.0:9000`)                 |
| `ui`       | Start the Gradio UI (default `0.0.0.0:7860`)                        |

### API endpoints

The backend exposes:

| Method | Path                 | Purpose                                                                  |
|--------|----------------------|--------------------------------------------------------------------------|
| `POST` | `/api/sessions`      | Register the current worker — body: `{"worker_host": "...", "worker_port": 9000}` |
| `GET`  | `/api/worker/info`   | Ask the active worker which model it's serving                           |
| `WS`   | `/api/generate`      | Streaming generation endpoint consumed by the UI                         |

### Output format

`generations.jsonl` — one JSON object per line with prompt, completion, metrics, and label:

```json
{
  "prompt_id": "tatsu-lab_alpaca-train-0",
  "prompt": "Give three tips for staying healthy.",
  "source_dataset": "tatsu-lab/alpaca",
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "generated_text": "1. Eat a balanced diet...",
  "degenerating": false,
  "chunk_summary": {
    "num_tokens": 256,
    "metrics_by_n": {
      "1": {"max_repetition": 0.18, "mean_ttr": 0.82},
      "3": {"max_repetition": 0.03, "mean_ttr": 0.97}
    }
  }
}
```

A generation is labelled `degenerating: true` when `max_repetition > 0.9` for unigrams.

### Metrics

- **TTR (Type-Token Ratio):** `unique_ngrams / total_ngrams`. 1.0 = fully diverse, 0.0 = fully repetitive.
- **Repetition:** `1 − TTR`.
- Computed over non-overlapping chunks of `chunk_size` tokens (default 256).
- Tracked for `n = 1` (unigrams) and `n = 3` (trigrams) by default.

### How probe training works

1. The base LLM is loaded frozen — no gradient updates flow to it.
2. A forward hook captures hidden states from the chosen layer during teacher-forced forward passes.
3. Hidden states are pooled over completion tokens (`mean`, `max`, or `last_token`).
4. A single `nn.Linear(hidden_size, 1)` head is trained with binary cross-entropy loss.
5. Evaluation reports accuracy, precision, recall, F1, AUC, and optimal threshold.
