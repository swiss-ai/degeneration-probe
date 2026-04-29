# Demo probe

A trained degeneration probe for **Apertus-8B-Instruct-2509**, committed to the repo so teammates can run the live UI without retraining.

## What's in here

| File | What |
|---|---|
| `adapter_model.safetensors` | LoRA weights (1.25 M trainable params on layer 16) |
| `adapter_config.json` | PEFT config |
| `probe_head.bin` | Value head — `Linear(4096 → 1)` |
| `probe_config.json` | Probe layer index and hidden size |
| `degeneration_meta.json` | model_name, window_size, primary_n, ttr_threshold |
| `training_config.json` | Full training config snapshot |

Total: ~4.8 MB.

## How it was trained

- **Base**: `swiss-ai/Apertus-8B-Instruct-2509` (frozen, 32 layers, hidden 4096)
- **Probe layer**: 31 (last) — value head reads the final hidden state
- **LoRA**: layer 16 only, rank 16, alpha 32
- **Target**: binary, `1 if TTR(next 256 bigrams) ≤ 0.2 else 0`
- **Loss**: BCE-with-logits
- **Data**: 30 % shuffled subsample (20 004 rows) of `luca-sartori/degeneration-probe-instruct` (gated)
- **Training**: 3 epochs, batch size 4, head LR 1e-3, LoRA LR 1e-5, seed 42
- **Hardware**: 1 × NVIDIA GH200 (CSCS Clariden)

## Final eval metrics

| | |
|---|---|
| MSE | 0.034 |
| Pearson | 0.865 |
| AUC @ 0.5 | 0.971 |

## How to use

Local worker (needs the gated Apertus-8B-Instruct on your HF account):

```bash
uv run python -m degeneration_probe worker --probe demo_probe --dtype bfloat16
```

The worker reads `degeneration_meta.json` to load the right base model automatically.

Cluster (CSCS Clariden):

```bash
PROBE=demo_probe ./cluster/start.sh
```

W&B run for this checkpoint: <https://wandb.ai/moritz-k-reihs-personal/Degeneration%20probe/runs/t18vdknx>
