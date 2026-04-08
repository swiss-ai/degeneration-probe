# degeneration-probe

A research pipeline for collecting labelled LLM generation data to train a **degeneration probe** — a lightweight classifier that detects repetitive model outputs from internal hidden states, without needing to re-run the full metrics pipeline at inference time.

The pipeline has three steps:
1. **Generate**: fetch prompts from a HuggingFace dataset, run an LLM to produce completions, and score each completion for repetition using chunk-level n-gram TTR metrics
2. **Train**: attach a lightweight linear probe to a hidden layer of the same LLM and train it to predict whether a generation is degenerating, using the labelled data from step 1
3. **Evaluate**: assess probe quality on held-out data (AUC, F1, ROC curve, threshold analysis)

All three steps are run through a single CLI: `python -m degeneration_probe {generate,train,evaluate}`.

---

## Prerequisites

- **Python 3.10–3.12**
- **[uv](https://docs.astral.sh/uv/)** (Python package manager)
- **A HuggingFace account** with an [access token](https://huggingface.co/settings/tokens) (needed to download models and datasets)
- **GPU recommended** for generation and training. The small Qwen-0.5B model runs on CPU/MPS but is slow.

---

## Setup

```bash
git clone <repo-url>
cd degeneration-probe

# Install all dependencies
uv sync

# Save your HuggingFace token (the pipeline reads it from here automatically)
mkdir -p keys
printf '%s' 'hf_YOUR_TOKEN_HERE' > keys/.hf_token
```

---

## Step 1: Generate labelled data

This step downloads prompts from a HuggingFace dataset, generates LLM completions, computes repetition metrics per completion, and labels each one as `degenerating: true/false`.

```bash
uv run python -m degeneration_probe generate --config configs/generate/alpaca_qwen05b.yaml
```

This uses the config at `configs/generate/alpaca_qwen05b.yaml`, which generates 10 completions from **Qwen-0.5B** on prompts from the **Alpaca** dataset. You can look at this file to see all available options (model, dataset, generation parameters, etc.).

When the run finishes, it prints the output directory:

```
Run complete. Results saved to: outputs/generations/20260403_115426
```

Inside that directory you'll find:
- `data/generations.jsonl` — the labelled data (this is what you need for training)
- `data/prompts.jsonl` — the raw prompts that were fetched
- `plots/` — histograms of TTR and repetition metrics
- `config.json` — exact config snapshot for reproducibility

---

## Step 2: Train a probe

Before training, you need to tell the training config where your generated data is. Open `configs/train/default.yaml` and update the `train_data` path to point to the `generations.jsonl` file from step 1:

```yaml
train_data:
  - outputs/generations/20260403_115426/data/generations.jsonl
```

You can also list multiple JSONL files here if you ran generation multiple times (e.g., with different datasets). All files must have been generated with the **same model** — this is validated automatically.

Then run:

```bash
uv run python -m degeneration_probe train --config configs/train/default.yaml
```

**Note:** you don't need to specify which model to use. The training script reads the `model_name` field from the JSONL data and loads the same model automatically. This prevents accidentally training a probe on representations from a different model than the one that generated the data.

When training finishes, it prints the output directory (e.g., `outputs/probes/20260403_115847`). Inside you'll find:
- `checkpoint/` — the trained probe weights (`probe_head.pt`) and config
- `eval/` — evaluation metrics, ROC curve plot, and threshold analysis plot
- `config.json` — full training config snapshot (including the auto-resolved model name)

Key training config options (in `configs/train/default.yaml`):
- `probe.layer`: which transformer layer to probe (`-1` = last layer)
- `probe.pooling`: how to aggregate hidden states over tokens (`mean`, `max`, or `last_token`)
- `num_epochs`, `learning_rate`, `batch_size`: standard training hyperparameters
- `eval_fraction`: fraction of data to hold out for evaluation (default: 0.2). Set `eval_data` to a separate JSONL path if you want to evaluate on entirely different data.

---

## Step 3: Evaluate a probe on new data

To evaluate a trained probe on a different dataset:

```bash
uv run python -m degeneration_probe evaluate \
  --checkpoint outputs/probes/20260403_115847/checkpoint \
  --eval_data outputs/generations/20260403_120000/data/generations.jsonl
```

The model is loaded automatically from the saved training config. Results (metrics JSON, ROC curve, threshold plot) are saved to the checkpoint's `eval/` subdirectory by default, or to a custom location with `--output_dir`.

Additional options:
- `--batch_size N` (default: 4)
- `--max_length N` (default: 2048)
- `--model_name NAME` — override the model (only needed if the saved config is missing)
- `--output_dir PATH` — save results to a custom directory instead of `checkpoint/eval/`

---

## Creating your own experiment configs

### Generation config

1. Copy an existing config:
   ```bash
   cp configs/generate/alpaca_qwen05b.yaml configs/generate/my_experiment.yaml
   ```

2. Edit the model and dataset sections:
   ```yaml
   model:
     name: Qwen/Qwen2.5-7B-Instruct    # any HuggingFace causal LM
     dtype: bfloat16                      # bfloat16 (GPU), float32 (CPU/MPS), or auto
     max_new_tokens: 512                  # how many tokens to generate per prompt
     temperature: 0.7
     top_p: 0.9

   dataset:
     name: openai/gsm8k                  # any HuggingFace dataset
     subset: main                         # dataset subset (null if none)
     split: test
     prompt_field: question               # which field contains the prompt (null = auto-detect)
     max_samples: 100                     # how many prompts to download
     max_prompts: 50                      # how many to actually generate for

   analysis:
     chunk_size: 256                      # tokens per chunk for metrics
     n_values: [1, 3]                     # n-gram sizes for TTR computation
   ```

3. Run:
   ```bash
   uv run python -m degeneration_probe generate --config configs/generate/my_experiment.yaml
   ```

### Training config

1. Copy the default:
   ```bash
   cp configs/train/default.yaml configs/train/my_training.yaml
   ```

2. Update `train_data` and any hyperparameters you want to change:
   ```yaml
   probe:
     layer: -1              # transformer layer to probe (-1 = last)
     pooling: mean           # mean | max | last_token
     threshold: 0.5          # classification threshold

   train_data:
     - outputs/generations/20260401_120000/data/generations.jsonl

   eval_fraction: 0.2        # hold-out fraction (ignored if eval_data is set)
   learning_rate: 1.0e-3
   batch_size: 4
   num_epochs: 10
   ```

3. Train:
   ```bash
   uv run python -m degeneration_probe train --config configs/train/my_training.yaml
   ```

For local-only configs you don't want to commit to git, name the file `*.local.yaml` — it will be gitignored automatically.

---

## Using multiple datasets

You may want to train a probe on data from multiple HuggingFace datasets (e.g., Alpaca + AIME) to make it more robust. Here's the workflow:

### 1. Generate data separately for each dataset

Each generation run uses one dataset and one model. Run the generate step once per dataset:

```bash
# Generate with Alpaca prompts
uv run python -m degeneration_probe generate --config configs/generate/alpaca_qwen05b.yaml
# -> outputs/generations/20260403_100000/data/generations.jsonl

# Generate with AIME prompts (same model, different dataset)
uv run python -m degeneration_probe generate --config configs/generate/aime_qwen05b.yaml
# -> outputs/generations/20260403_100500/data/generations.jsonl
```

**Important:** all generation configs must use the **same model** (e.g., `Qwen/Qwen2.5-0.5B-Instruct`). The training step validates this and will refuse to proceed if the data files come from different models.

### 2. List all data files in the training config

Point `train_data` to all the JSONL files you want to combine:

```yaml
train_data:
  - outputs/generations/20260403_100000/data/generations.jsonl   # Alpaca
  - outputs/generations/20260403_100500/data/generations.jsonl   # AIME
```

The training script concatenates all records into a single dataset. The train/eval split (controlled by `eval_fraction`) is applied to the combined data.

### 3. Train as usual

```bash
uv run python -m degeneration_probe train --config configs/train/my_training.yaml
```

### 4. Evaluate on specific datasets

To see how the probe performs on a specific dataset, pass that dataset's JSONL to the evaluate command:

```bash
# Evaluate on Alpaca data only
uv run python -m degeneration_probe evaluate \
  --checkpoint outputs/probes/20260403_110000/checkpoint \
  --eval_data outputs/generations/20260403_100000/data/generations.jsonl \
  --output_dir outputs/probes/20260403_110000/eval_alpaca

# Evaluate on AIME data only
uv run python -m degeneration_probe evaluate \
  --checkpoint outputs/probes/20260403_110000/checkpoint \
  --eval_data outputs/generations/20260403_100500/data/generations.jsonl \
  --output_dir outputs/probes/20260403_110000/eval_aime
```

Use `--output_dir` to save each evaluation to a separate directory so results don't overwrite each other.

---

## Dataset Caching

Downloaded prompt samples are cached in `data/cache/` (gitignored) so subsequent runs with the same config skip the HuggingFace download entirely. The cache filename encodes all parameters that affect the sample, so changing any of them produces a new cache file automatically.

To override the cache location (e.g., on a shared cluster filesystem):

```bash
export PROBE_DATA_CACHE=/path/to/shared/cache
```

---

## Interactive Visualization & Steering

The project includes a real-time web interface for observing degeneration as it happens and steering the model away from repetitive loops.

The system has three components that run as separate processes:

| Component | Command | Default Port | Purpose |
|-----------|---------|-------------|---------|
| **Backend** | `python -m degeneration_probe serve` | 8000 | FastAPI server — REST API + WebSocket relay |
| **Worker** | `python -m degeneration_probe worker` | 9000 | Loads the model + probe, generates tokens |
| **UI** | `python -m degeneration_probe ui` | 7861 | Gradio web interface |

### Quick start (local, no GPU needed)

Open three terminal tabs:

```bash
# Tab 1: Start the backend
uv run python -m degeneration_probe serve

# Tab 2: Start the worker with a small model
uv run python -m degeneration_probe worker --model Qwen/Qwen2.5-0.5B-Instruct --dtype float32

# Tab 3: Start the UI
uv run python -m degeneration_probe ui --port 7861
```

Open **http://localhost:7861** in your browser. Click **Connect** (defaults to localhost:9000), type a prompt, and click **Generate**.

### Using a trained probe

If you have a trained probe checkpoint from the training pipeline, pass it to the worker:

```bash
uv run python -m degeneration_probe worker \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --probe outputs/probes/<timestamp>/checkpoint \
  --dtype float32
```

The probe scores each token as it is generated. Scores are shown as color-coded tokens in the UI (green = safe, amber = borderline, red = degenerate).

### Steering

Enable steering in the UI sidebar to let the probe intervene during generation. When the probe score exceeds the threshold, the selected strategy modifies the model's output distribution to break out of repetitive loops.

Available strategies:
- **Temperature Boost** — divides logits by a higher temperature when degeneration is detected, increasing output diversity

### Running with Apertus 8B on Clariden

On the GPU node (via `salloc` or SLURM job):

```bash
uv run python -m degeneration_probe worker \
  --model Swiss-AI/Apertus-8B-Instruct \
  --probe outputs/probes/<timestamp>/checkpoint \
  --port 9000
```

On your local machine, set up an SSH tunnel and run the backend + UI:

```bash
# Tunnel the worker port
ssh -L 9000:localhost:9000 <user>@<clariden-node>

# Local terminals
uv run python -m degeneration_probe serve
uv run python -m degeneration_probe ui --port 7861
```

The UI connects to `localhost:9000` which tunnels to the GPU worker.

### API Endpoints

The backend exposes a REST API for programmatic access:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/strategies` | List available steering strategies |
| `POST` | `/api/sessions` | Connect to a worker (`{"worker_host": "...", "worker_port": ...}`) |
| `GET` | `/api/sessions/current` | Current connection status |
| `DELETE` | `/api/sessions/current` | Disconnect from worker |
| `GET` | `/api/generations` | List past generations |
| `GET` | `/api/generations/{id}` | Get a generation with per-token probe scores |
| `WS` | `/api/generate` | WebSocket endpoint for streaming generation |

---

## Running on the ETH Euler Cluster

### One-time setup

```bash
ssh <nethz>@euler.ethz.ch
git clone <repo-url> ~/degeneration-probe
cd ~/degeneration-probe
bash cluster/setup_euler.sh

# Save your HuggingFace token
mkdir -p keys
printf '%s' 'hf_YOUR_TOKEN_HERE' > ~/degeneration-probe/keys/.hf_token
chmod 600 ~/degeneration-probe/keys/.hf_token
```

### Submitting a generation job

```bash
# Default config (Alpaca + Qwen-0.5B)
sbatch cluster/euler_generate.sh

# Custom config
GEN_CONFIG=configs/generate/aime_apertus8b.yaml sbatch cluster/euler_generate.sh
```

Logs go to `logs/generate_<jobid>.out` and `logs/generate_<jobid>.err`.

| Resource | Default |
|----------|---------|
| Time limit | 4 hours |
| CPUs | 4 |
| RAM | 32 GB (4 x 8 GB) |
| GPU | 1 x 40 GB VRAM |

### Submitting a training job

```bash
# Default config
sbatch cluster/euler_train.sh

# Custom config
TRAIN_CONFIG=configs/train/my_training.yaml sbatch cluster/euler_train.sh
```

Logs go to `logs/train_<jobid>.out` and `logs/train_<jobid>.err`.

| Resource | Default |
|----------|---------|
| Time limit | 8 hours |
| CPUs | 4 |
| RAM | 32 GB (4 x 8 GB) |
| GPU | 1 x 40 GB VRAM |

### Checking job status

```bash
squeue -u $USER                    # list running/pending jobs
cat logs/generate_<jobid>.out      # check generation output
cat logs/train_<jobid>.out         # check training output
```

Adjust `#SBATCH` directives in `cluster/euler_generate.sh` or `cluster/euler_train.sh` for different needs.

---

## Output Format

### `generations.jsonl`

Each line is a JSON object with the prompt, completion, metrics, and label:

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

---

## Metrics

- **TTR (Type-Token Ratio):** `unique_ngrams / total_ngrams`. 1.0 = fully diverse, 0.0 = fully repetitive.
- **Repetition:** `1 - TTR`.
- Computed over non-overlapping chunks of `chunk_size` tokens (default 256).
- Tracked for `n=1` (unigrams) and `n=3` (trigrams) by default.

---

## How Probe Training Works

1. The base LLM is loaded frozen (no gradient updates to the LLM)
2. A forward hook captures hidden states from the chosen layer during teacher-forced forward passes
3. Hidden states are pooled over completion tokens (mean, max, or last-token pooling)
4. A single `nn.Linear(hidden_size, 1)` head is trained with binary cross-entropy loss
5. Evaluation reports accuracy, precision, recall, F1, AUC, and optimal threshold
