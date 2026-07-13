"""Subtask F: shared evaluation harness for the trained onset probes (`probe_N` /
multi-horizon), one layer at a time.

Loads the probes saved by `scripts/train_onset_probes.py` for a given `--layer`
(`<output_root>/probes/onset/layer_<NN>/{probe_5,...,multi_horizon}/`) and, for
each of `val` / `test_indomain` / `test_heldout_domains`, computes:

1. Per-horizon, per-design (`probe_N` vs `multi_horizon`) AUC / average precision
   / Brier score -- `eval_metrics.parquet`. Note `test_heldout_domains` has only
   11 positive rollouts total (see the implementation prompt's known limitation);
   its numbers are reported but are low-confidence by construction, not a bug.
2. `probe_N`-vs-`multi_horizon` agreement, per horizon: Pearson correlation of
   logits, Spearman correlation of ranks, and 0.5-threshold decision agreement
   rate -- does the independently-trained single-horizon head roughly agree with
   the corresponding head of the jointly-trained multi-horizon model?
   -- `agreement.parquet`.
3. Head-nesting internal-consistency check: since horizon labels nest by
   construction (every token positive for `y_5` is also positive for `y_20`,
   `y_50`, ... -- see `onset_dataset.OnsetActivationDataset._labels_for`), a
   sane model's predicted positive rate should be non-decreasing in horizon
   (`rate(head_50) >= rate(head_20) >= head_5`, etc.), and most *individual*
   tokens should have `prob(head_N) <= prob(head_{N'})` for `N < N'`. Reports
   the per-horizon mean predicted probability (should be monotonic in horizon)
   and the fraction of tokens that violate that pairwise ordering for each
   adjacent horizon pair -- `nesting.parquet`.

This is eval-only: it does NOT re-materialize the `train` split (no training
happens here), only `val`/`test_indomain`/`test_heldout_domains`, each read from
disk exactly once via `onset_dataset.materialize_features` (same one-shot,
no-repeated-IO approach `train_onset_probes.py` uses).
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from degeneration_probe.data.onset_dataset import build_rollout_index, materialize_features
from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.dataset_gen.onset_labels import DEFAULT_HORIZONS
from degeneration_probe.probes.onset_probe import OnsetProbeHead

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"
DEFAULT_SPLITS = ("val", "test_indomain", "test_heldout_domains")
EVAL_NEGATIVE_TOKENS_PER_ROLLOUT = 32


def load_probes(layer_dir: Path, horizons: List[int]) -> Dict[str, OnsetProbeHead]:
    probes = {f"probe_{h}": OnsetProbeHead.load(layer_dir / f"probe_{h}") for h in horizons}
    probes["multi_horizon"] = OnsetProbeHead.load(layer_dir / "multi_horizon")
    for m in probes.values():
        m.eval()
    return probes


def predict_probs(probes: Dict[str, OnsetProbeHead], X: torch.Tensor, horizons: List[int]) -> Dict[str, np.ndarray]:
    """horizon -> {'probe_N': probs[N], 'multi_horizon': probs[N]}, sigmoid-ed."""
    with torch.no_grad():
        probe_n_probs = {h: torch.sigmoid(probes[f"probe_{h}"](X)[:, 0]).numpy() for h in horizons}
        mh_probs = torch.sigmoid(probes["multi_horizon"](X)).numpy()  # [N, K]
    out = {}
    for i, h in enumerate(horizons):
        out[h] = {"probe_N": probe_n_probs[h], "multi_horizon": mh_probs[:, i]}
    return out


def compute_metrics_rows(layer: int, split: str, y_np: np.ndarray, probs_by_horizon: dict, horizons: List[int]) -> List[dict]:
    rows = []
    for i, horizon in enumerate(horizons):
        y_true = y_np[:, i]
        has_both_classes = len(np.unique(y_true)) > 1
        for design in ("probe_N", "multi_horizon"):
            p = probs_by_horizon[horizon][design]
            auc = roc_auc_score(y_true, p) if has_both_classes else float("nan")
            ap = average_precision_score(y_true, p) if has_both_classes else float("nan")
            brier = brier_score_loss(y_true, p)
            rows.append(
                {
                    "layer": layer, "split": split, "horizon": horizon, "design": design,
                    "auc": auc, "ap": ap, "brier": brier, "n": len(y_true), "n_pos": int(y_true.sum()),
                }
            )
    return rows


def compute_agreement_rows(layer: int, split: str, probs_by_horizon: dict, horizons: List[int]) -> List[dict]:
    rows = []
    for horizon in horizons:
        a = probs_by_horizon[horizon]["probe_N"]
        b = probs_by_horizon[horizon]["multi_horizon"]
        pearson = pearsonr(a, b).statistic if len(a) > 1 else float("nan")
        spearman = spearmanr(a, b).statistic if len(a) > 1 else float("nan")
        agreement_rate = float(np.mean((a >= 0.5) == (b >= 0.5)))
        rows.append(
            {
                "layer": layer, "split": split, "horizon": horizon,
                "pearson_corr": pearson, "spearman_corr": spearman, "decision_agreement_rate_0.5": agreement_rate,
            }
        )
    return rows


def compute_nesting_rows(layer: int, split: str, probs_by_horizon: dict, horizons: List[int]) -> List[dict]:
    """Horizons nest (y_5 => y_20 => ... => y_200 by construction), so mean predicted
    probability should be non-decreasing in horizon, and few tokens should have
    prob(smaller horizon) > prob(larger horizon). Checked separately for probe_N and
    multi_horizon, since only the latter is guaranteed anything about cross-head
    consistency by its joint training -- probe_N heads are fully independent models
    and could violate this much more freely."""
    sorted_horizons = sorted(horizons)
    rows = []
    for design in ("probe_N", "multi_horizon"):
        mean_probs = {h: float(np.mean(probs_by_horizon[h][design])) for h in sorted_horizons}
        for h_small, h_large in combinations(sorted_horizons, 2):
            p_small = probs_by_horizon[h_small][design]
            p_large = probs_by_horizon[h_large][design]
            violation_rate = float(np.mean(p_small > p_large + 1e-6))
            rows.append(
                {
                    "layer": layer, "split": split, "design": design,
                    "horizon_small": h_small, "horizon_large": h_large,
                    "mean_prob_small": mean_probs[h_small], "mean_prob_large": mean_probs[h_large],
                    "mean_prob_monotonic": mean_probs[h_small] <= mean_probs[h_large],
                    "pairwise_violation_rate": violation_rate,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--splits", nargs="*", type=str, default=list(DEFAULT_SPLITS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = DatasetGenConfig.from_yaml(args.config)
    onset_labels_df = pd.read_parquet(paths.onset_labels_path(config))
    layer_dir = paths.dataset_root(config) / "probes" / "onset" / f"layer_{args.layer:02d}"

    probes = load_probes(layer_dir, args.horizons)

    metrics_rows, agreement_rows, nesting_rows = [], [], []
    for split in args.splits:
        split_index = build_rollout_index(onset_labels_df, split)
        X, Y = materialize_features(
            config, split_index, layer=args.layer, horizons=args.horizons,
            max_negative_tokens_per_rollout=EVAL_NEGATIVE_TOKENS_PER_ROLLOUT,
            negative_rollouts_per_positive=None,  # eval: every rollout in the split, not a subsample
            seed=args.seed,
        )
        print(f"[layer {args.layer}] {split}: materialized {tuple(X.shape)}")

        probs_by_horizon = predict_probs(probes, X, args.horizons)
        y_np = Y.numpy()

        metrics_rows += compute_metrics_rows(args.layer, split, y_np, probs_by_horizon, args.horizons)
        agreement_rows += compute_agreement_rows(args.layer, split, probs_by_horizon, args.horizons)
        nesting_rows += compute_nesting_rows(args.layer, split, probs_by_horizon, args.horizons)

    metrics_df = pd.DataFrame.from_records(metrics_rows)
    agreement_df = pd.DataFrame.from_records(agreement_rows)
    nesting_df = pd.DataFrame.from_records(nesting_rows)

    metrics_df.to_parquet(layer_dir / "eval_metrics.parquet", index=False)
    agreement_df.to_parquet(layer_dir / "agreement.parquet", index=False)
    nesting_df.to_parquet(layer_dir / "nesting.parquet", index=False)

    print(metrics_df.to_string(index=False))
    print(agreement_df.to_string(index=False))
    print(nesting_df.to_string(index=False))
    print(f"[layer {args.layer}] wrote eval_metrics.parquet, agreement.parquet, nesting.parquet to {layer_dir}")


if __name__ == "__main__":
    main()
