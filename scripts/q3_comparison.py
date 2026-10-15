"""Every scorer of the Q3 comparison on one table: references, baselines and the probes.

One row per (model, split, budget, scorer), with the seed mean and standard
deviation of the realised false-alarm rate on the split, the detection rate,
AUROC, post-onset coverage, warning coverage, the coverage of the text before
the warning window, the early-warning rate and the lead.

- Random alarms and the token-cap rule follow in closed form from the onset
  labels. Random alarms flag each token with a probability q calibrated on the
  healthy validation responses; their response score is the maximum of T
  independent uniform scores, so a degenerate response outranks a healthy one
  with probability T_d / (T_d + T_h), which gives their AUROC. The first alarm
  of a response is a geometric variable cut at its length, so the median lead
  is read off the pooled distribution of first alarm minus onset. The token-cap rule flags only the last
  token of a capped response, and its threshold flags nothing on validation,
  where no healthy response is capped.
- Deterministic baselines are read from outputs/baselines/<model>/<name>, with
  the n of the repeated-n-gram rule chosen on validation.
- Probes are read from selected_metrics.csv at the reported layers: the frozen
  probe, the adapted probe at the frozen probe's layer and the adapted probe at
  its own selected layer.

    python scripts/q3_comparison.py

Writes <outputs>/analysis/q3_comparison.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_selected import read_depth  # noqa: E402
from scripts.random_alarm_reference import UNJUDGED, calibrate  # noqa: E402

STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
OUTPUTS = STORE / "outputs"
MODELS = {"apertus": "apertus-8b-instruct", "llama3p1": "llama3p1-8b-instruct", "mistral": "mistral-7b-instruct-v0p1"}
NGRAM = {"apertus": "ngram_30", "llama3p1": "ngram_8", "mistral": "ngram_16"}
BASELINES = {"repetition_trailing": "windowed seq-rep-2", "rep_l": "rep/l", "entropy": "windowed entropy", "neglogp": "windowed -log p"}
LAYERS = {"apertus": (15, 30), "llama3p1": (15, 6), "mistral": (5, 11)}
SPLITS = ["test_indomain", "test_heldout_domains"]
BUDGETS = [0.01, 0.05, 0.10]
RECIPE = "frontier_window W=512 centered K=256"
METRICS = ["negative_fpr", "detection_rate", "rollout_auc", "in_pattern_recall", "warning_recall_256",
           "preonset_coverage_256", "fired_before_frontier", "median_offset"]


def labels(dataset: str) -> pd.DataFrame:
    frame = pd.read_parquet(STORE / f"degeneration-dataset-{dataset}" / "onset_labels" / "onset_labels.parquet")
    return frame[~frame["onset_resolution"].isin(UNJUDGED)]


def random_alarms(frame: pd.DataFrame, split: str, budget: float) -> dict:
    q = calibrate(frame[(frame["split"] == "val") & ~frame["is_positive"]]["num_tokens"].to_numpy(float), budget)
    part = frame[frame["split"] == split]
    deg, healthy = part[part["is_positive"]], part[~part["is_positive"]]
    lengths, onsets = deg["num_tokens"].to_numpy(float), deg["onset_position"].to_numpy(float)
    healthy_lengths = healthy["num_tokens"].to_numpy(float)
    alarm = 1 - (1 - q) ** lengths
    auc = np.mean(lengths[:, None] / (lengths[:, None] + healthy_lengths[None, :]))
    # Expected number of first alarms at or before onset + x, summed over degenerate responses.
    grid = np.arange(-int(onsets.max()), int(lengths.max()))
    reached = np.clip(grid[:, None] + onsets[None, :] + 1, 0, lengths[None, :])
    pooled = (1 - (1 - q) ** reached).sum(axis=1)
    return {"negative_fpr": np.mean(1 - (1 - q) ** healthy_lengths), "detection_rate": alarm.mean(),
            "rollout_auc": auc, "in_pattern_recall": q, "warning_recall_256": q, "preonset_coverage_256": q,
            "fired_before_frontier": (1 - (1 - q) ** onsets).sum() / alarm.sum(),
            "median_offset": float(grid[np.searchsorted(pooled, alarm.sum() / 2)]) if len(deg) else np.nan}


def token_cap(frame: pd.DataFrame, split: str) -> dict:
    part = frame[frame["split"] == split]
    deg, healthy = part[part["is_positive"]], part[~part["is_positive"]]
    capped_healthy = (healthy["stop_reason"] != "eos").mean()
    after = (deg["num_tokens"] - deg["onset_position"]).to_numpy(float)
    return {"negative_fpr": capped_healthy, "detection_rate": 1.0, "rollout_auc": 1 - capped_healthy / 2,
            "in_pattern_recall": len(deg) / after.sum(), "warning_recall_256": 0.0, "preonset_coverage_256": 0.0,
            "fired_before_frontier": 0.0, "median_offset": float(np.median(after - 1))}


def from_evaluation(evaluation: Path, split: str, budget: float) -> dict | None:
    frame = read_depth(evaluation, split)
    if frame is None:
        return None
    row = frame[np.isclose(frame["target_negative_fpr"], budget)].iloc[0]
    out = {metric: row[metric] for metric in METRICS if metric in row}
    out["detection_rate"] = row["detected_positives"] / row["positives"]
    return out


def main() -> None:
    probes = pd.read_csv(OUTPUTS / "analysis" / "selected_metrics.csv")
    probes = probes[probes["recipe"].str.startswith(RECIPE)]
    probes["detection_rate"] = probes["detected_positives"] / probes["positives"]
    records = []
    for model, dataset in MODELS.items():
        frame = labels(dataset)
        frozen_layer, adapted_layer = LAYERS[model]
        for split in SPLITS:
            for budget in BUDGETS:
                key = {"model": model, "split": split, "budget": budget}
                records.append({**key, "scorer": "random alarms", "seed": 0, **random_alarms(frame, split, budget)})
                records.append({**key, "scorer": "token-cap rule", "seed": 0, **token_cap(frame, split)})
                for name, label in [(NGRAM[model], f"repeated-n-gram rule (n={NGRAM[model].split('_')[1]})"), *BASELINES.items()]:
                    values = from_evaluation(OUTPUTS / "baselines" / dataset / name / "evaluation", split, budget)
                    if values is not None:
                        records.append({**key, "scorer": label, "seed": 0, **values})
                for lora, layer, label in [("none", frozen_layer, f"frozen probe (L{frozen_layer})"),
                                           ("all", frozen_layer, f"adapted probe (L{frozen_layer})"),
                                           ("all", adapted_layer, f"adapted probe (L{adapted_layer})")]:
                    part = probes[(probes["model"] == model) & (probes["lora"] == lora) & (probes["layer"] == layer)
                                  & (probes["split"] == split) & np.isclose(probes["target_negative_fpr"], budget)]
                    for seed, (_, row) in enumerate(part.iterrows()):
                        records.append({**key, "scorer": label, "seed": seed, **{m: row[m] for m in METRICS}})
    raw = pd.DataFrame(records)
    grouped = raw.groupby(["model", "split", "budget", "scorer"], sort=False)
    table = grouped[METRICS].mean().join(grouped[METRICS].std().add_suffix("_std")).join(grouped.size().rename("seeds"))
    out = OUTPUTS / "analysis" / "q3_comparison.csv"
    table.reset_index().to_csv(out, index=False)
    print(f"{len(table)} rows -> {out}")


if __name__ == "__main__":
    main()
