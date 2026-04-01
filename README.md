# degeneration-demo

Standalone local demo for:

- loading a small Hugging Face instruct model
- generating a completion from a prompt
- scoring repetition-based degeneration on the generated tokens
- extending that check to chunk-level `n=1` / `n=3` analysis over many prompts

## Setup

```bash
cd degeneration-demo
uv sync
```

Create a local token file:

```bash
mkdir -p keys
printf '%s\n' 'hf_...' > keys/.hf_token
```

## Run

```bash
uv run python scripts/demo_degeneration.py \
  --device cpu \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --torch_dtype float32 \
  --max_new_tokens 64
```

## Fetch Prompt Sample

```bash
uv run python scripts/fetch_prompts.py \
  --dataset tatsu-lab/alpaca \
  --split train \
  --max_samples 25
```

This writes:

```text
outputs/prompts_sample.jsonl
```

## Run On Many Prompts
```bash
uv run python scripts/run_prompt_batch.py \
  --input_path outputs/prompts_sample.jsonl \
  --output_path outputs/labeled_generations.jsonl \
  --max_prompts 10 \
  --device cpu \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --torch_dtype float32 \
  --max_new_tokens 256
```

This writes:

```text
outputs/labeled_generations.jsonl
```

Each record includes the prompt, generated completion, and a `chunk_summary` with non-overlapping chunk metrics for `n=1` and `n=3`.

## Plot Metric Distributions

```bash
uv run python scripts/plot_degeneration_metrics.py \
  --input_path outputs/labeled_generations.jsonl
```

This writes histogram plots to:

```text
outputs/plots/
```

## Run Full Pipeline

```bash
uv run python scripts/main.py \
  --dataset tatsu-lab/alpaca \
  --split train \
  --max_samples 10 \
  --max_prompts 5 \
  --device cpu \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --torch_dtype float32 \
  --max_new_tokens 256
```

This single entrypoint:

- fetches a prompt sample from Hugging Face
- runs batch generation on those prompts
- saves chunk-based statistics
- saves histogram plots for the chosen `n` values
