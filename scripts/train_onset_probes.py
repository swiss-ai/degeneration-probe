"""Subtasks D+E: train `probe_N` (5 independent per-horizon heads) and the
multi-horizon head, both at a single layer, from cached activations -- no-LoRA
path.

Trains all 6 heads together for one `--layer` to share the (expensive)
activation-reading pass: activations are read from disk exactly ONCE per split
(`degeneration_probe.data.onset_dataset.materialize_features`), producing
in-memory `(X, Y)` tensors; every subsequent training epoch is a pure in-memory
tensor op. This matters a lot at this corpus's scale -- re-reading via a naive
per-epoch `DataLoader` loop would reopen thousands of safetensors files on every
epoch for identical data, multiplying wall-clock time by the epoch count for no
benefit (see the module docstring in `onset_dataset.materialize_features`).

Per the 2026-07-12 plan change, this is meant to be run once per layer (e.g. via
`cluster/train_onset_probes.sbatch`'s array over layer 0..32) rather than at a
single "best" layer -- the probe heads are tiny and every layer's activations are
already cached, so there's no compute reason to pick a winner ahead of time.

`probe_N`: 5 independent `OnsetProbeHead(hidden, 1)` models, each with its own
Adam optimizer, each trained only on its own horizon's masked BCE loss (see
`compute_probe_bce_loss` in `degeneration_probe/training/loss.py` -- this script
inlines the same masked-BCE-with-pos-weight idea directly on flat tensors rather
than importing that function, since there's no padding/ignore-label masking
needed here: every row in the materialized `Y` is already a valid label).

Multi-horizon: one shared `OnsetProbeHead(hidden, K)`, one optimizer, loss is a
weighted sum of the K per-horizon BCE terms (inverse-positive-rate weights,
renormalized to sum to K -- see the implementation prompt's recommendation to
ablate this against uniform weighting later).

Writes, under `<output_root>/probes/onset/layer_<NN>/`:
    probe_<N>/{probe_head.bin, probe_config.json}   for each horizon N
    multi_horizon/{probe_head.bin, probe_config.json}
    eval_val.parquet   -- per-horizon, per-design AUC/AP on the val split
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from degeneration_probe.data.onset_dataset import (
    build_rollout_index,
    materialize_features,
    pos_weight_for_horizons,
)
from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.dataset_gen.onset_labels import DEFAULT_HORIZONS
from degeneration_probe.probes.onset_probe import OnsetProbeHead

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"
HIDDEN_SIZE = 4096


def iterate_minibatches(n: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def evaluate(
    models: Dict[str, OnsetProbeHead], X: torch.Tensor, Y: torch.Tensor, horizons: List[int], device: torch.device
) -> pd.DataFrame:
    for m in models.values():
        m.eval()
    with torch.no_grad():
        probe_n_logits = {h: models[f"probe_{h}"](X.to(device)).cpu().numpy()[:, 0] for h in horizons}
        mh_logits = models["multi_horizon"](X.to(device)).cpu().numpy()  # [N, K]

    y_np = Y.numpy()
    rows = []
    for i, horizon in enumerate(horizons):
        y_true = y_np[:, i]
        has_both_classes = len(np.unique(y_true)) > 1
        for design, logits in [("probe_N", probe_n_logits[horizon]), ("multi_horizon", mh_logits[:, i])]:
            auc = roc_auc_score(y_true, logits) if has_both_classes else float("nan")
            ap = average_precision_score(y_true, logits) if has_both_classes else float("nan")
            rows.append(
                {"horizon": horizon, "design": design, "auc": auc, "ap": ap, "n": len(y_true), "n_pos": int(y_true.sum())}
            )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--mlp-hidden-dim", type=int, default=None, help="Unset = plain linear probe.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096, help="Token-level minibatch size.")
    parser.add_argument("--max-negative-tokens-per-rollout", type=int, default=32)
    parser.add_argument("--negative-rollouts-per-positive", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = DatasetGenConfig.from_yaml(args.config)
    onset_labels_df = pd.read_parquet(paths.onset_labels_path(config))
    train_index = build_rollout_index(onset_labels_df, "train")
    val_index = build_rollout_index(onset_labels_df, "val")

    t0 = time.time()
    X_train, Y_train = materialize_features(
        config, train_index, layer=args.layer, horizons=args.horizons,
        max_negative_tokens_per_rollout=args.max_negative_tokens_per_rollout,
        negative_rollouts_per_positive=args.negative_rollouts_per_positive,
        seed=args.seed,
    )
    X_val, Y_val = materialize_features(
        config, val_index, layer=args.layer, horizons=args.horizons,
        max_negative_tokens_per_rollout=args.max_negative_tokens_per_rollout,
        negative_rollouts_per_positive=None,  # eval set: every negative rollout, not just a subsample
        seed=args.seed,
    )
    print(
        f"[layer {args.layer}] materialized train={tuple(X_train.shape)} val={tuple(X_val.shape)} "
        f"in {time.time() - t0:.1f}s"
    )
    X_train, Y_train = X_train.to(device), Y_train.to(device)

    pos_weight_by_horizon = pos_weight_for_horizons(train_index, horizons=args.horizons)
    print(f"[layer {args.layer}] pos_weight per horizon (corpus-true, train split): {pos_weight_by_horizon}")
    raw_head_weights = np.array([pos_weight_by_horizon[h] for h in args.horizons], dtype=np.float32)
    head_weights = raw_head_weights / raw_head_weights.sum() * len(args.horizons)
    pos_weight_tensors = {h: torch.tensor(pos_weight_by_horizon[h], device=device) for h in args.horizons}

    models = {f"probe_{h}": OnsetProbeHead(HIDDEN_SIZE, 1, args.mlp_hidden_dim).to(device) for h in args.horizons}
    models["multi_horizon"] = OnsetProbeHead(HIDDEN_SIZE, len(args.horizons), args.mlp_hidden_dim).to(device)
    optimizers = {k: torch.optim.Adam(m.parameters(), lr=args.lr) for k, m in models.items()}

    n_train = X_train.shape[0]
    t0 = time.time()
    for epoch in range(args.epochs):
        for m in models.values():
            m.train()
        epoch_losses = {k: 0.0 for k in models}
        n_batches = 0
        for batch_idx in iterate_minibatches(n_train, args.batch_size, rng):
            batch_idx_t = torch.as_tensor(batch_idx, device=device)
            x = X_train[batch_idx_t]
            y = Y_train[batch_idx_t]
            n_batches += 1

            for i, horizon in enumerate(args.horizons):
                opt = optimizers[f"probe_{horizon}"]
                opt.zero_grad()
                logits = models[f"probe_{horizon}"](x)[:, 0]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y[:, i], pos_weight=pos_weight_tensors[horizon]
                )
                loss.backward()
                opt.step()
                epoch_losses[f"probe_{horizon}"] += loss.item()

            opt = optimizers["multi_horizon"]
            opt.zero_grad()
            logits = models["multi_horizon"](x)
            per_head_losses = [
                nn.functional.binary_cross_entropy_with_logits(
                    logits[:, i], y[:, i], pos_weight=pos_weight_tensors[horizon]
                )
                for i, horizon in enumerate(args.horizons)
            ]
            loss = sum(w * l for w, l in zip(head_weights, per_head_losses))
            loss.backward()
            opt.step()
            epoch_losses["multi_horizon"] += loss.item()

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            avg = {k: v / n_batches for k, v in epoch_losses.items()}
            print(f"[layer {args.layer}] epoch {epoch + 1}/{args.epochs} avg loss: {avg}")

    print(f"[layer {args.layer}] training took {time.time() - t0:.1f}s ({n_train} train tokens)")

    eval_df = evaluate(models, X_val, Y_val, args.horizons, device)
    eval_df.insert(0, "layer", args.layer)
    print(eval_df.to_string(index=False))

    out_dir = Path(args.out_dir) if args.out_dir else paths.dataset_root(config) / "probes" / "onset" / f"layer_{args.layer:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, model in models.items():
        model.save(out_dir / key)
    eval_df.to_parquet(out_dir / "eval_val.parquet", index=False)
    print(f"[layer {args.layer}] saved probes + eval to {out_dir}")


if __name__ == "__main__":
    main()
