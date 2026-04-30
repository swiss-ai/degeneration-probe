# Ablation experiments — degeneration probe

A running plan for one-thing-at-a-time experiments against the current
baseline checkpoint. Each ablation gets its own training config, its own
W&B run, and a one-paragraph postmortem appended below.

## Baseline (the bar to beat)

Reference checkpoint: `outputs/probes/20260429_001809/` (also shipped as `demo_probe/`).

| | |
|---|---|
| Model | swiss-ai/Apertus-8B-Instruct-2509 (frozen, 32 layers, hidden 4096) |
| Probe layer | 31 (last) |
| LoRA | rank 16, alpha 32, attached only to layer 16 |
| Label | binary point label per token: `1 if TTR(tokens[t : t+256], n=2) ≤ 0.2 else 0` |
| Loss | masked BCE-with-logits |
| Data | 30 % shuffled subsample (20 004 / 833 rows) of `luca-sartori/degeneration-probe-instruct` |
| Training | 3 epochs, batch_size 4, head LR 1e-3, LoRA LR 1e-5, seed 42 |
| Final eval | **MSE 0.034 · Pearson 0.865 · AUC@0.5 0.971** |
| W&B run | <https://wandb.ai/moritz-k-reihs-personal/Degeneration%20probe/runs/t18vdknx> |

## Planned ablations

| # | Name | Hypothesis | Status |
|---|---|---|---|
| 1 | Smoothed window-average labels | Replacing the binary point label with the *fraction* of overlapping windows that are degenerate gives a smoother, less brittle target. | implemented (`feature/smoothed-labels`), awaiting cluster run |
| 2 | LoRA layer sweep (probe @ 0.95·N) | A single-layer LoRA placed in different positions of the stack has very different "reach" — we want to find the sweet spot for adapting representations that the late-layer probe reads. | planned |

Each ablation changes exactly one thing vs the baseline; everything else (model, probe layer, LoRA layer, data subsample seed, eval split, hyperparameters) stays fixed so the comparison is clean.

---

## Ablation 1 — Smoothed binary labels (windowed average)

### Current behaviour

For each completion token at position `t`, the target is

```
target_point[t] = 1 if TTR(tokens[t : t+256], n=2) ≤ 0.2 else 0
```

i.e. exactly one forward window starting at `t`. Tokens within the last 256 of the completion are masked (no full forward window fits).

### Proposed behaviour

Token `t` belongs to *every* size-256 window whose start index `s` satisfies `s ≤ t ≤ s+255`. The valid range, clipped to the completion `[0, L)`, is:

```
s_lo = max(0, t - 255)
s_hi = min(t, L - 256)         # window must fit
```

If `s_lo > s_hi` the token has no valid window → masked, same as current behaviour.

For each valid start `s` define the binary window-degeneracy

```
D(s) = 1 if TTR(tokens[s : s+256], n=2) ≤ 0.2 else 0
```

and

```
target_smoothed[t] = mean over s ∈ [s_lo, s_hi] of D(s)
```

Targets are now continuous in `[0, 1]` rather than binary.

For internal tokens (deep enough in the completion that 256 windows cover them) this is a 256-wide moving average over the array `D[0..L-256]`.

### Why this might help

- **Sharpness penalty.** Two adjacent tokens with nearly-identical hidden states can carry opposite labels under the point scheme — that's a hard target. The smoothed label changes gradually with position.
- **Boundary localisation.** Smoothed values carry information about *how close* a token is to a degenerate region, which is exactly the forecasting signal a steering probe wants.
- **Useful gradient on imbalanced batches.** Most tokens in the dataset are negative (clean). With binary labels, batches dominated by clean tokens give the optimiser little signal. Continuous targets always carry gradient.

### Why it might *not* help

- Smoothing might dilute the signal: a token sitting right outside a tight loop will get a small positive label, and the probe might learn "small but nonzero everywhere" instead of separating clean from degenerate.
- Eval AUC@0.5 is harder to interpret — the probe's predicted score is no longer aimed at a 0/1 target. We may need to re-binarise the smoothed target at 0.5 to compute AUC, which throws away the calibration information we just added.

### Implementation

**Config** (our `src/degeneration_probe/config.py`):

- Add `LabelConfig.smoothing: bool = False`. When `true`, the label collation produces continuous targets and the trainer uses MSE.

**Fork — `degeneration/data_loader.py`**:

- `make_collate_fn` grows a `smoothing: bool = False` kwarg.
- When `smoothing=True`, after computing `rep[s] = 1 - TTR(window starting at s)` for all `s` in the completion, build the binary indicator array `D[s] = 1.0 if rep[s] >= 1 - ttr_threshold else 0.0` (only for positions where the window fits). Then compute `target[t] = mean(D[s] for s in [max(0, t-255), min(t, L-256)])` and set `label_mask[t] = 1` whenever that range is non-empty.
- Keep the existing branches (continuous `1 - TTR` regression; binary point label) intact.

**Fork — `degeneration/train.py`**:

- `TrainConfig.label_smoothing: Optional[bool] = None` — pass through to collate.
- Loss selection: `BCE` when `ttr_threshold is set AND not smoothing`; otherwise `MSE` on `sigmoid(logit)`.
- Record `label_smoothing` in `degeneration_meta.json` so the worker / evaluator can re-derive the same labels at eval time.

**Fork — `degeneration/evaluate.py`**:

- `evaluate_checkpoint` reads `label_smoothing` from meta and forwards it to `make_collate_fn`.

**Our `__main__.py::cmd_train`**: plumb `cfg.label.smoothing` → `TrainConfig.label_smoothing`.

**New config**: `configs/train/apertus8b_hf_smoothed.yaml` — a copy of `apertus8b_hf.yaml` with `label.smoothing: true`. Same data subsample seed, same epochs, same LRs, same LoRA layout, same probe layer.

### Verification

1. Local smoke run on `_smoke_train_hf.local.yaml` with `label.smoothing: true` — confirm the trainer loads, no shape errors.
2. Inspect a single batch: `print(labels.unique()[:10], labels.min(), labels.mean(), labels.max())` should show fractional values, not just `{0.0, 1.0}`.
3. Compare per-token target distributions: histogram of point-labels vs smoothed-labels on the same 100 completions. Smoothed should be roughly the convolution of the point distribution with a 256-wide box.
4. Cluster training run with the new config. Monitor W&B for: train/mse curve smoothness vs the baseline's train/bce, eval/mse, eval/pearson, eval/auc_at_0_5.

### Comparison protocol

Both runs use the same eval split (833 rows, seed 42 shuffle). Report:

| Metric | Baseline | Smoothed | Verdict |
|---|---|---|---|
| Eval MSE | 0.034 | ? | |
| Eval Pearson | 0.865 | ? | |
| Eval AUC@0.5 | 0.971 | ?* | |
| Worker UI subjective | sharp 0/1 colours | gradient colours | |

*For the AUC, re-binarise the smoothed target at 0.5 (a token whose mean window-degeneracy ≥ 0.5) so the comparison is apples-to-apples.

### Outcomes & decision rule

- **Pearson improves** AND AUC stays within 1 pp of baseline → smoothed becomes the new default. Keep both code paths but flip `LabelConfig.smoothing` default to `true`.
- **Pearson is similar** but AUC drops noticeably → keep both options, document that "binary point label is sharper for the steering use case, smoothed is calibrated for analysis."
- **Both metrics worse** → revert; document the negative result in this file and move to ablation #2.

### Postmortem

(filled in after the run)

---

## Ablation 2 — LoRA layer sweep (probe fixed at 0.95·N)

### Question

Where in the stack does the LoRA adapter need to sit for the late-layer probe to read the cleanest signal? The baseline put LoRA on layer 16 (mid-stack) and the probe on layer 31 (last). We don't know whether mid-stack is actually the sweet spot or whether earlier / later positions would do better. Sweeping rules that out.

### Setup

| | |
|---|---|
| **Probe layer** | `floor(0.95 · num_layers)` — for Apertus-8B's 32 layers that's layer **30** (was 31 in the baseline; 0.95 instead of "last" so we have one layer of headroom for LoRA at the probe layer itself if we ever want it). |
| **LoRA layers** | sweep over `[0, 2, 4, …, 30]` — single-layer LoRA at each even index from 0 up to and including the probe layer. **16 runs total.** |
| **Everything else** | identical to the baseline: rank 16 / alpha 32 / no dropout, ttr_threshold 0.2 (binary BCE — *not* the smoothed labels of ablation 1, this sweep tests the LoRA position alone), 30% data subsample (20004 / 833 rows, seed 42), 3 epochs, batch_size 4, head LR 1e-3, LoRA LR 1e-5. |

LoRA at layers > probe_layer is ruled out by construction (gradient wouldn't flow there).

### Why this question matters

- Our first attempt (single-layer LoRA at the probe layer itself, with probe at layer 31) failed to learn — pearson 0, AUC 0.5.
- Our successful run (LoRA at layer 16, probe at layer 31) hit AUC 0.97.
- That's two data points on the same axis with wildly different outcomes. Sweeping fills in the curve.
- A clean answer here is reusable for every future probe we train on this model: "use a single rank-16 LoRA at layer X for best results."

### Implementation

**Approach: generate one config per layer, submit them as separate sbatch jobs.** Each run logs to W&B with a distinguishing name so they group together as a sweep.

**File layout:**

```
configs/train/sweeps/lora_layer_sweep/
  lora_l00.yaml
  lora_l02.yaml
  ...
  lora_l30.yaml
```

Each is a copy of `configs/train/apertus8b_hf.yaml` with two overrides:
- `probe.layer: 30` (was -1)
- `probe.lora.layers: [<N>]`
- `wandb_run_name: "apertus8b-lora-l<NN>"`

**Generation script** `scripts/generate_lora_sweep_configs.py`: takes the baseline path + a list of LoRA layers + the probe layer, writes the configs. Idempotent (overwrite on re-run).

**Submission script** `cluster/submit_sweep.sh <configs-dir>`: loops the dir, submits one sbatch per config via `sbatch cluster/clariden_train.sh <path>`, prints a job-id list. Writes the job IDs to `.run/sweep_jobids.txt` so we can monitor.

**Tagging in W&B:** every run in the sweep gets `wandb_run_name` set; the W&B UI lets us group by name pattern.

### Compute budget

16 runs × ~30 min per run on a single GH200 = ~8 GPU-hours total. The runs are independent so they can fan out across however many GPUs the queue gives us. Realistic wall-clock if Clariden's `debug` partition has slots: a few hours.

### Comparison protocol

After all 16 runs finish, plot `eval/auc_at_0_5` vs LoRA layer index (16 points). Same for `eval/pearson` and `eval/mse`. We expect a non-monotonic curve — the question is where it peaks.

Decision rule:
- **Clear peak at some layer L\*** → that becomes the new default for `probe.lora.layers` in `apertus8b_hf.yaml`. Update the README and the demo probe to match.
- **Roughly flat curve** → LoRA position doesn't matter much; pick any (probably the cheapest = mid-stack) and document.
- **Monotonic (e.g. always-better closer to probe)** → consider a *range* of layers near the probe (e.g. `[28, 29, 30]`) as the new default; that's a follow-up ablation.

### Don't

- Don't combine with smoothed labels in this sweep — confounds two variables.
- Don't sweep odd layers — step of 2 is enough resolution; we can fill in odd layers later if a peak is sharp.
- Don't change rank/alpha/LRs — those are separate ablations.

### Postmortem

(filled in after the runs)
