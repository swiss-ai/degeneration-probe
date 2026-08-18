"""Fill in coverage at every warning width for a run already replayed.

A replayed run records coverage of the run-up at whichever widths were being
reported when it ran. Adding a width later does not need the replay repeated:
the threshold each checkpoint was measured at is fixed by the healthy answers,
whose peak scores were kept, and coverage of the approach is a statement about
the degenerate answers alone. So only those have to be scored again, which is a
hundred-odd answers rather than three and a half thousand.

The recovered thresholds are checked against the recorded ones before anything
is written. They are derived from the same numbers by the same rule, so they
must agree exactly; if they do not, something about the run has moved and its
existing columns and its new ones would describe different scorers.

    python scripts/backfill_warning_bands.py --run-dir outputs/<run>/latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import LabelConfig
from degeneration_probe.data.activation_store import load_probe_layers
from degeneration_probe.data.dataset import load_degeneration_records
from degeneration_probe.evaluation.head_selection import STEERING_BUDGET
from degeneration_probe.evaluation.protocol import (
    WARNING_BANDS,
    coverage_window,
    threshold_for_budget,
)
from replay_checkpoints import (
    check_collapsible,
    collapse_heads,
    load_run_config,
    score_rollout,
)

REPLAY_FILE = "checkpoint_replay.parquet"
THRESHOLD_FILE = "checkpoint_replay_thresholds.npz"


def recovered_thresholds(peaks: np.ndarray, budget: float) -> np.ndarray:
    """The threshold every checkpoint of every depth was measured at."""
    checkpoints, depths = peaks.shape[1], peaks.shape[2]
    taus = np.zeros((checkpoints, depths), dtype=np.float64)
    for checkpoint in range(checkpoints):
        for depth in range(depths):
            taus[checkpoint, depth], _ = threshold_for_budget(
                peaks[:, checkpoint, depth].astype(np.float64), budget
            )
    return taus


def check_against_record(frame: pd.DataFrame, taus, steps, layers) -> None:
    recorded = frame.set_index(["step", "layer"])["budget_tau"]
    for checkpoint, step in enumerate(steps):
        for depth, layer in enumerate(layers):
            was = float(recorded.loc[(int(step), int(layer))])
            now = float(taus[checkpoint, depth])
            if not np.isclose(was, now, rtol=0, atol=1e-9):
                raise SystemExit(
                    f"step {int(step)} depth {int(layer)} was measured at {was!r} and the "
                    f"saved healthy answers give {now!r}. The recorded columns and any new "
                    "ones would describe different scorers, so nothing was written."
                )


@torch.no_grad()
def coverage(run_dir: Path, taus, steps, layers, device) -> pd.DataFrame:
    """Coverage at every width, over the degenerate answers only."""
    config = load_run_config(run_dir)
    config.dataset.sampling.evaluation_negative_rollouts_per_positive = None
    records = [
        record
        for record in load_degeneration_records(
            config.dataset,
            split=config.dataset.splits.validation,
            label_config=LabelConfig(),
            training=False,
        )
        if record.is_positive
    ]
    print(f"{len(records)} degenerate answers over {len(steps)} checkpoints x {len(layers)} depths")

    weights, biases = collapse_heads(run_dir, steps, layers)
    weights, biases = weights.to(device), biases.to(device)
    threshold = torch.as_tensor(taus, dtype=torch.float32, device=device)

    shape = (len(steps), len(layers))
    hits = {band: torch.zeros(shape, dtype=torch.float64, device=device) for band in WARNING_BANDS}
    totals: Dict[int, int] = {band: 0 for band in WARNING_BANDS}
    inside_hits = torch.zeros(shape, dtype=torch.float64, device=device)
    inside_total = 0

    for index, record in enumerate(records):
        features = load_probe_layers(
            config.dataset.build_root,
            record.domain,
            record.prompt_id,
            record.rollout_idx,
            probe_layers=list(layers),
        )
        length = min(features.shape[1], len(record.targets))
        # [checkpoints, depths, tokens]
        scores = score_rollout(features[:, :length].to(device), weights, biases)
        onset = int(record.onset_position)
        above = scores >= threshold.unsqueeze(-1)
        window = coverage_window(length, onset, None)
        inside_hits += above[:, :, window].sum(dim=2).double()
        inside_total += window.stop - window.start
        for band in WARNING_BANDS:
            run_up = coverage_window(length, onset, band)
            hits[band] += above[:, :, run_up].sum(dim=2).double()
            totals[band] += run_up.stop - run_up.start
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(records)} answers", flush=True)

    rows = []
    for checkpoint, step in enumerate(steps):
        for depth, layer in enumerate(layers):
            row = {"step": int(step), "layer": int(layer)}
            row["in_pattern_recall_check"] = (
                float(inside_hits[checkpoint, depth]) / inside_total if inside_total else np.nan
            )
            for band in WARNING_BANDS:
                row[f"warning_recall_{band}"] = (
                    float(hits[band][checkpoint, depth]) / totals[band] if totals[band] else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    frame = pd.read_parquet(run_dir / REPLAY_FILE)
    material = np.load(run_dir / THRESHOLD_FILE)
    peaks, steps, layers = material["peaks"], material["steps"], material["layers"]
    check_collapsible(load_run_config(run_dir))

    taus = recovered_thresholds(peaks, STEERING_BUDGET)
    check_against_record(frame, taus, steps, layers)
    print(f"thresholds for {len(steps)} checkpoints x {len(layers)} depths match the record")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    measured = coverage(run_dir, taus, steps, layers, device)

    # Coverage inside the loop is already recorded, so recomputing it is a check
    # that the answers scored here are the answers scored then.
    merged = frame.merge(measured, on=["step", "layer"], how="left", suffixes=("_old", ""))
    drift = (merged["in_pattern_recall"] - merged["in_pattern_recall_check"]).abs().max()
    if drift > 1e-6:
        raise SystemExit(
            f"coverage inside the loop moved by {drift:.3g} against the recorded value, so the "
            "answers scored now are not the answers scored then. Nothing was written."
        )
    merged = merged.drop(columns=["in_pattern_recall_check"])
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_old")])
    merged.to_parquet(run_dir / REPLAY_FILE, index=False)
    widths = ", ".join(str(band) for band in WARNING_BANDS)
    print(f"coverage at widths {widths} -> {run_dir / REPLAY_FILE}")


if __name__ == "__main__":
    main()
