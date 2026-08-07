# Degeneration probe

This repository trains one per-token task, `degeneration`, on the materialized
`degeneration-dataset-apertus-8b-instruct` build. The probe is a scalar linear
layer attached to Apertus layer 30, with optional LoRA adapters.

## Configuration

Hydra composes three independent groups from `configs/main.yaml`:

- `configs/model/apertus.yaml`: model, tokenizer and model dtype;
- `configs/training/degeneration.yaml`: probe, LoRA, loss, optimizer, runtime,
  validation, checkpoints and W&B;
- `configs/dataset/degeneration-dataset-apertus-8b-instruct.yaml`: materialized
  build path, splits, tokenization and sampling.

The generation recipes for all `degeneration-dataset-*` builds are kept
separately under `configs/dataset/builds/`. Training still defaults to the
materialized Apertus-8B-Instruct build.

## Training

The default loss is BCE. For a rollout with a defined `onset_position`, tokens
before onset are `0` and tokens at or after onset are `1`. EOS rollouts are all
zero; truncated rollouts without a defined onset are excluded.

```bash
uv run python scripts/train_probe.py
```

Use MSE against the original per-token `repetition_score` with:

```bash
uv run python scripts/train_probe.py training.loss.name=mse
```

All W&B settings are under `training.wandb`; disable logging with
`training.wandb.enabled=false`.

Validation currently reports loss, valid-token count and basic target/prediction
statistics. `ValidationMetric` provides an empty registry for future metrics.

## Dataset pipeline

Rollout generation, repetition labeling, onset materialization, LLM judging and
activation caching remain independent pipeline stages under
`degeneration_probe/dataset_gen/`. Their default CLI build config is the file in
`configs/dataset/builds/`.

## Tests

```bash
uv run pytest
```
