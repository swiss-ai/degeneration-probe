# degeneration-probe

A research pipeline for collecting labelled LLM generation data to train a **degeneration probe** — a lightweight classifier that detects repetitive model outputs from internal hidden states, without needing to re-run the full metrics pipeline at inference time.

The current pipeline:
1. Fetches prompts from a HuggingFace dataset
2. Runs an LLM to generate completions
3. Scores each completion for repetition using chunk-level n-gram TTR metrics
4. Saves results and histogram plots

**Phase 2** adds probe training: a linear classifier on frozen LLM hidden states that learns to predict the `degenerating` label.

---

## Quick Start (Local)

```bash
git clone <repo-url>
cd degeneration-probe

# Install dependencies (requires Python 3.10–3.12 and uv)
uv sync

# Place your HuggingFace token
mkdir -p keys
printf '%s' 'hf_YOUR_TOKEN_HERE' > keys/.hf_token

# Run the full pipeline with the default small config
uv run python run.py --config configs/alpaca_qwen05b.yaml
```

Results appear under `outputs/runs/<timestamp>/`.

---

## Project Structure

```
degeneration-probe/
│
├── run.py                        # Single entry point — runs the full pipeline
│
├── configs/                      # Experiment configs (version-controlled YAML)
│   ├── alpaca_qwen05b.yaml       # Small local test: Alpaca dataset + Qwen-0.5B
│   └── aime_qwen05b.yaml         # AIME math problems + Qwen-0.5B
│
├── scripts/
│   ├── train.py                  # Hydra entry point for probe training
│   └── evaluate.py               # Standalone probe evaluation
│
├── src/degeneration_probe/       # Core library
│   ├── config.py                 # Config dataclasses (Phase 1 + Phase 2)
│   ├── data.py                   # Prompt fetching and JSONL I/O
│   ├── generation.py             # LLM generation + batch processing
│   ├── metrics.py                # n-gram TTR / repetition metrics
│   ├── plotting.py               # Histogram plots
│   ├── representations.py        # Hidden-states extractor (stub — in development)
│   ├── model_utils.py            # Model/tokenizer loading helpers
│   ├── types.py                  # Data types (DegenerationItem)
│   ├── dataset.py                # PyTorch Dataset + collation for probe training
│   ├── probe.py                  # SequenceProbe (hook → pool → linear head)
│   ├── training.py               # Training loop
│   ├── evaluation.py             # Evaluation pipeline + metrics/plots
│   └── utils/
│       ├── hooks.py              # Forward hook context manager
│       └── clf_metrics.py        # Classification metrics + ROC plotting
│
├── cluster/
│   ├── setup_euler.sh            # One-time setup on ETH Euler login node
│   └── euler_run.sh              # SLURM job script
│
├── keys/
│   └── .hf_token                 # Your HuggingFace token (gitignored)
│
├── data/                         # Dataset prompt cache (gitignored, shared across runs)
│   └── cache/
│       └── tatsu-lab__alpaca_none_train_s42_n25.jsonl
│
└── outputs/                      # All run outputs (gitignored)
    └── runs/
        └── 20260401_143022/      # One folder per run (timestamp)
            ├── config.json       # Exact config used for reproducibility
            ├── data/
            │   ├── prompts.jsonl
            │   └── generations.jsonl
            └── plots/
                ├── ttr_n1.png
                ├── repetition_n1.png
                ├── ttr_n3.png
                └── repetition_n3.png
```

---

## Dataset Caching

Downloaded prompt samples are cached in `data/cache/` (gitignored) so subsequent runs with the same config skip the HuggingFace download entirely.

The cache filename encodes all parameters that affect the sample — dataset, subset, split, seed, size, and shuffle — so changing any of them produces a new cache file automatically.

To override the cache location (e.g. on a shared cluster filesystem):

```bash
export PROBE_DATA_CACHE=/path/to/shared/cache
uv run python run.py --config configs/my_experiment.yaml
```

On ETH Euler, `setup_euler.sh` sets `PROBE_DATA_CACHE` to scratch space automatically.

---

## Adding a New Experiment

1. Copy an existing config:
   ```bash
   cp configs/alpaca_qwen05b.yaml configs/my_experiment.yaml
   ```

2. Edit `model.name` and `dataset.name` (and `prompt_field` if the dataset needs it):
   ```yaml
   model:
     name: Qwen/Qwen2.5-7B-Instruct
     dtype: bfloat16
     max_new_tokens: 512
   dataset:
     name: openai/gsm8k
     subset: main
     split: test
     prompt_field: question
     max_samples: 100
     max_prompts: 50
   ```

3. Run:
   ```bash
   uv run python run.py --config configs/my_experiment.yaml
   ```

For local-only tweaks you don't want to commit, name the file `*.local.yaml` — it will be gitignored automatically.

---

## Running on the ETH Euler Cluster

### One-time setup (do this once per account)

```bash
ssh <nethz>@euler.ethz.ch
git clone <repo-url> ~/degeneration-probe
cd ~/degeneration-probe
bash cluster/setup_euler.sh
```

Place your HuggingFace token on the cluster:
```bash
printf '%s' 'hf_YOUR_TOKEN_HERE' > ~/degeneration-probe/keys/.hf_token
chmod 600 ~/degeneration-probe/keys/.hf_token
```

### Submitting a job

```bash
# Default config (alpaca_qwen05b.yaml)
sbatch cluster/euler_run.sh

# Custom config
CONFIG_FILE=my_experiment.yaml sbatch cluster/euler_run.sh
```

Logs are written to `logs/run_<jobid>.out` and `logs/run_<jobid>.err`.
Results appear in `outputs/runs/<timestamp>/` on the cluster. Copy them back with `scp` or sync via git.

### SLURM defaults

| Resource | Default |
|----------|---------|
| Time limit | 4 hours |
| CPUs | 4 |
| RAM | 32 GB (4 × 8 GB) |
| GPU | 1 × ≥ 20 GB VRAM |

Adjust `#SBATCH` directives in `cluster/euler_run.sh` (data collection) or `cluster/euler_train.sh` (probe training) for larger models.

---

## Output Format

### `config.json`
A snapshot of every parameter used in the run — useful for reproducing or comparing results.

### `data/prompts.jsonl`
One record per prompt:
```json
{"prompt_id": "tatsu-lab_alpaca-train-0", "prompt": "...", "source_dataset": "tatsu-lab/alpaca"}
```

### `data/generations.jsonl`
One record per generation, with chunk-level degeneration metrics:
```json
{
  "prompt_id": "tatsu-lab_alpaca-train-0",
  "prompt": "...",
  "source_dataset": "tatsu-lab/alpaca",
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "generated_text": "...",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "chunk_summary": {
    "num_tokens": 256,
    "chunk_size": 256,
    "num_chunks": 1,
    "metrics_by_n": {
      "1": {"min_ttr": 0.82, "max_repetition": 0.18, "mean_ttr": 0.82, "mean_repetition": 0.18, "per_chunk": [...]},
      "3": {"min_ttr": 0.97, "max_repetition": 0.03, "mean_ttr": 0.97, "mean_repetition": 0.03, "per_chunk": [...]}
    }
  },
  "degenerating": false,
  "hidden_states_path": null
}
```

**`degenerating`** is `true` when `max_repetition > 0.9` for the first n-gram size — used as a binary label for probe training.

**`hidden_states_path`** is `null` until hidden-states extraction is implemented (see below).

---

## Metrics Explained

- **TTR (Type-Token Ratio):** `unique_ngrams / total_ngrams`. Score of 1.0 = fully diverse; 0.0 = fully repetitive.
- **Repetition:** `1 - TTR`.
- Computed over non-overlapping **chunks** of `chunk_size` tokens (default 256). The per-chunk breakdown shows *where* in the generation repetition occurs.
- Two n-gram sizes are tracked by default: `n=1` (unigrams) and `n=3` (trigrams).

---

## Probe Training (Phase 2)

Once you have labelled generation data (from Phase 1), train a degeneration probe:

### Local

```bash
# Train with the small Qwen model on synthetic test data (sanity check)
uv run python scripts/train.py

# Train with a specific dataset and model
uv run python scripts/train.py model=apertus train_data=outputs/runs/XXXXXXXX/data/generations.jsonl

# Override any parameter
uv run python scripts/train.py model=apertus num_epochs=20 probe.layer=28 probe.pooling=max
```

### Cluster (Euler)

```bash
# Default config
sbatch cluster/euler_train.sh

# With overrides
HYDRA_OVERRIDES="model=apertus train_data=/path/to/data.jsonl" sbatch cluster/euler_train.sh
```

### Evaluate a saved checkpoint

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/probes/<timestamp>/checkpoint \
  --eval_data path/to/eval.jsonl \
  --model_name "Qwen/Qwen2.5-0.5B-Instruct"
```

Results (metrics JSON, ROC curve, threshold analysis plot) are saved to the checkpoint's `eval/` subdirectory.

### How it works

1. The base LLM is loaded frozen (no gradient updates)
2. A forward hook captures hidden states from a chosen layer during teacher-forced forward passes over the labelled data
3. Hidden states are pooled over completion tokens (mean, max, or last-token)
4. A single `nn.Linear(hidden_size, 1)` head is trained with BCE loss
5. Evaluation reports accuracy, precision, recall, F1, AUC, and optimal threshold

---

## Coming Soon

- Multi-layer probing: train probes at every layer to find where degeneration is most detectable
- Pre-saved hidden states for faster probe iteration
- Interactive visualization demo
- Training-time steering integration
