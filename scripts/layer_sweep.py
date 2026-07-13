"""Subtask C: full per-layer, per-horizon onset-separability AUC sweep.

Redoes this project's earlier ad hoc, 300-sample, `lrs_first_start`-based
layer/horizon check properly: all 33 `hidden_states` layers, the full set of
train/val positive rollouts (not a capped sample), the actual 5 candidate probe
horizons {5, 20, 50, 100, 200}, and the *adopted*
`lrs_first_start_normalized_growing` onset positions (via
`degeneration_probe.dataset_gen.onset_labels.resolve_onset_position`, already
materialized by subtask A into `onset_labels/onset_labels.parquet`) -- not the
superseded plain-LRS numbers from the earlier session.

For each layer L and horizon N, fits a standardized logistic regression on
train-split examples and reports ROC AUC on val-split examples (clean
rollout-level train/val separation, no train-adjacent leakage):
    - positive example: `hidden_states[L, max(0, onset - N), :]`, one per
      train/val-split positive rollout (single fixed offset from onset, not the
      cumulative "within N of onset" window `probe_N` itself will train on --
      this sweep is purely about how separable a token *exactly* N tokens before
      onset is from a typical non-degenerating token, at each layer).
    - negative example: `hidden_states[L, random_position, :]`, one per sampled
      negative (EOS) rollout -- the same negative sample is reused across all 5
      horizons (only the positive side depends on N).

Meant to run via `cluster/layer_sweep.sbatch` (CPU-only -- no GPU/model forward
pass needed, this only reads already-cached activations off disk). Writes
`onset_labels/layer_sweep_auc.parquet` (columns: layer, horizon, auc, n_fit_pos,
n_fit_neg, n_eval_pos, n_eval_neg).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from safetensors.torch import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from degeneration_probe.data.onset_dataset import NUM_HIDDEN_STATE_LAYERS, build_rollout_index
from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.dataset_gen.onset_labels import DEFAULT_HORIZONS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"


def _read_all_layers_at_position(
    config: DatasetGenConfig, domain: str, prompt_id: str, rollout_idx: int, position: int
) -> np.ndarray:
    """One partial/mmap'd read covering every layer at a single token position --
    [NUM_HIDDEN_STATE_LAYERS, hidden_size]."""
    activation_path = paths.rollout_activation_path(config, domain, prompt_id, rollout_idx)
    with safe_open(str(activation_path), framework="pt") as f:
        vec = f.get_slice("hidden_states")[:, position, :]
    return vec.float().numpy()


def _sample_negative_rows(rng: np.random.Generator, negative_rows: pd.DataFrame, n: int) -> pd.DataFrame:
    n = min(n, len(negative_rows))
    chosen = rng.choice(len(negative_rows), size=n, replace=False)
    return negative_rows.iloc[chosen]


def _collect_negative_features(
    config: DatasetGenConfig, negative_rows: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """[n, NUM_HIDDEN_STATE_LAYERS, hidden_size], one random position per rollout."""
    feats = [
        _read_all_layers_at_position(
            config, row.domain, row.prompt_id, int(row.rollout_idx), int(rng.integers(0, int(row.num_tokens)))
        )
        for row in negative_rows.itertuples(index=False)
    ]
    return np.stack(feats)


def _collect_positive_features(
    config: DatasetGenConfig, positive_rows: pd.DataFrame, horizons: Sequence[int]
) -> Dict[int, np.ndarray]:
    """horizon -> [n, NUM_HIDDEN_STATE_LAYERS, hidden_size], at position max(0, onset - horizon)."""
    out: Dict[int, list] = {horizon: [] for horizon in horizons}
    for row in positive_rows.itertuples(index=False):
        onset = int(row.onset_position)
        for horizon in horizons:
            position = max(0, onset - horizon)
            out[horizon].append(
                _read_all_layers_at_position(config, row.domain, row.prompt_id, int(row.rollout_idx), position)
            )
    return {horizon: np.stack(v) for horizon, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument(
        "--negative-multiplier", type=float, default=5.0,
        help="Negative rollouts sampled per positive rollout, in each of the fit/eval splits (default: 5).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-lr-iter", type=int, default=500)
    args = parser.parse_args()

    config = DatasetGenConfig.from_yaml(args.config)
    onset_labels_df = pd.read_parquet(paths.onset_labels_path(config))
    rng = np.random.default_rng(args.seed)

    fit_index = build_rollout_index(onset_labels_df, "train")
    eval_index = build_rollout_index(onset_labels_df, "val")

    fit_pos = fit_index[fit_index.is_positive]
    eval_pos = eval_index[eval_index.is_positive]
    fit_neg_rows = _sample_negative_rows(
        rng, fit_index[~fit_index.is_positive], round(len(fit_pos) * args.negative_multiplier)
    )
    eval_neg_rows = _sample_negative_rows(
        rng, eval_index[~eval_index.is_positive], round(len(eval_pos) * args.negative_multiplier)
    )

    print(
        f"fit(train): {len(fit_pos)} pos / {len(fit_neg_rows)} neg; "
        f"eval(val): {len(eval_pos)} pos / {len(eval_neg_rows)} neg"
    )

    t0 = time.time()
    fit_neg_feats = _collect_negative_features(config, fit_neg_rows, rng)
    eval_neg_feats = _collect_negative_features(config, eval_neg_rows, rng)
    fit_pos_feats = _collect_positive_features(config, fit_pos, args.horizons)
    eval_pos_feats = _collect_positive_features(config, eval_pos, args.horizons)
    print(f"Read all activations in {time.time() - t0:.1f}s")

    y_fit = np.concatenate([np.ones(len(fit_pos)), np.zeros(len(fit_neg_rows))])
    y_eval = np.concatenate([np.ones(len(eval_pos)), np.zeros(len(eval_neg_rows))])

    results = []
    t0 = time.time()
    for layer in range(NUM_HIDDEN_STATE_LAYERS):
        for horizon in args.horizons:
            X_fit = np.concatenate([fit_pos_feats[horizon][:, layer, :], fit_neg_feats[:, layer, :]])
            X_eval = np.concatenate([eval_pos_feats[horizon][:, layer, :], eval_neg_feats[:, layer, :]])

            scaler = StandardScaler().fit(X_fit)
            clf = LogisticRegression(max_iter=args.max_lr_iter).fit(scaler.transform(X_fit), y_fit)
            auc = roc_auc_score(y_eval, clf.decision_function(scaler.transform(X_eval)))

            results.append(
                {
                    "layer": layer,
                    "horizon": horizon,
                    "auc": auc,
                    "n_fit_pos": len(fit_pos),
                    "n_fit_neg": len(fit_neg_rows),
                    "n_eval_pos": len(eval_pos),
                    "n_eval_neg": len(eval_neg_rows),
                }
            )
            print(f"layer={layer:2d} horizon={horizon:3d} auc={auc:.4f}")
    print(f"Fit+eval all (layer, horizon) pairs in {time.time() - t0:.1f}s")

    results_df = pd.DataFrame.from_records(results)
    out_path = paths.onset_labels_dir(config) / "layer_sweep_auc.parquet"
    results_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(results_df)} rows to {out_path}")

    pivot = results_df.pivot(index="layer", columns="horizon", values="auc")
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
