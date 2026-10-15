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
  the n of the repeated-n-gram rule chosen on validation. With --sweep, every
  baseline is read instead at the setting scripts/select_baseline_sweep.py
  chose on validation, from outputs/baselines_sweep/<model>/<setting>.
- Probes are read from selected_metrics.csv at the reported layers: the frozen
  probe, the adapted probe at the frozen probe's layer and the adapted probe at
  its own selected layer.
- With --human every scorer is read on the human-verified test splits instead:
  the onset labels take the reviewers' verdicts and onsets from
  <store>/human_review/labels/labels_<dataset>.csv, and baselines and probes are
  read from the ``evaluation_human`` folder that scripts/evaluate_human_labels.py
  writes next to each ``evaluation`` folder. Thresholds stay those frozen on
  validation.

    python scripts/q3_comparison.py [--sweep] [--human]

Writes <outputs>/analysis/q3_comparison[_sweep][_human].csv.
"""

from __future__ import annotations

import argparse
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
HUMAN_LABELS = STORE / "human_review" / "labels"
METRICS = ["negative_fpr", "detection_rate", "rollout_auc", "in_pattern_recall", "warning_recall_256",
           "preonset_coverage_256", "fired_before_frontier", "median_offset"]


def labels(dataset: str, human: bool = False) -> pd.DataFrame:
    frame = pd.read_parquet(STORE / f"degeneration-dataset-{dataset}" / "onset_labels" / "onset_labels.parquet")
    frame = frame[~frame["onset_resolution"].isin(UNJUDGED)]
    if not human:
        return frame
    reviewed = pd.read_csv(HUMAN_LABELS / f"labels_{dataset}.csv").set_index("id")
    keys = frame["prompt_id"] + "::" + frame["rollout_idx"].astype(str)
    frame = frame[~keys.isin(reviewed.index[reviewed["drop"]])].copy()
    keys = frame["prompt_id"] + "::" + frame["rollout_idx"].astype(str)
    hit = keys.isin(reviewed.index).to_numpy()
    frame.loc[hit, "is_positive"] = reviewed.loc[keys[hit], "is_positive"].to_numpy(bool)
    frame.loc[hit, "onset_position"] = reviewed.loc[keys[hit], "onset_position"].to_numpy(float)
    return frame


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


def baseline_dirs(model: str, dataset: str, sweep: bool, evaluation: str = "evaluation") -> list[tuple[Path, str]]:
    """The evaluation directory and row label of every deterministic baseline of one model."""
    if not sweep:
        return [(OUTPUTS / "baselines" / dataset / name / evaluation, label)
                for name, label in [(NGRAM[model], f"repeated-n-gram rule (n={NGRAM[model].split('_')[1]})"),
                                    *BASELINES.items()]]
    chosen = pd.read_csv(OUTPUTS / "analysis" / "baseline_sweep_selected.csv")
    chosen = chosen[(chosen["model"] == model) & (chosen["role"] == "selected")]
    labels = {"ngram": "repeated-n-gram rule", "repetition_trailing": "windowed seq-rep-n", "rep_l": "rep/l",
              "entropy": "windowed entropy", "neglogp": "windowed -log p"}
    settings = chosen.drop_duplicates("family").set_index("family")["setting"]
    return [(OUTPUTS / "baselines_sweep" / dataset / settings[family] / evaluation,
             f"{label} ({settings[family]})" if family == "ngram" else label)
            for family, label in labels.items()]


def human_probes(probes: pd.DataFrame) -> pd.DataFrame:
    """The reported probes' rows of selected_metrics.csv with their test-split metrics replaced by the
    human-verified ones in each depth's ``evaluation_human`` folder. A probe whose folder is missing keeps no
    test rows, so it shows up as absent instead of silently carrying judge labels."""
    wanted = []
    for model, (frozen_layer, adapted_layer) in LAYERS.items():
        part = probes[probes["model"] == model]
        wanted.append(part[((part["lora"] == "none") & (part["layer"] == frozen_layer))
                           | ((part["lora"] == "all") & part["layer"].isin([frozen_layer, adapted_layer]))])
    rows = []
    for _, run in pd.concat(wanted).drop_duplicates(["run_name", "attempt", "layer"]).iterrows():
        folder = OUTPUTS / run["run_name"] / str(run["attempt"]) / "selected" / f"layer_{int(run['layer']):02d}"
        for split in SPLITS:
            frame = read_depth(folder / "evaluation_human", split)
            if frame is None:
                print(f"  no human evaluation: {folder.relative_to(OUTPUTS)} {split}")
                continue
            for _, metrics in frame.iterrows():
                rows.append({**{k: run[k] for k in ("model", "lora", "layer", "recipe", "seed", "run_name", "attempt")},
                             **metrics.to_dict(), "split": split})
    out = pd.DataFrame(rows)
    out["detection_rate"] = out["detected_positives"] / out["positives"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true", help="read each baseline at its setting chosen on validation")
    parser.add_argument("--human", action="store_true", help="read every scorer on the human-verified test splits")
    args = parser.parse_args()
    evaluation = "evaluation_human" if args.human else "evaluation"
    probes = pd.read_csv(OUTPUTS / "analysis" / "selected_metrics.csv")
    probes = probes[probes["recipe"].str.startswith(RECIPE)]
    probes["detection_rate"] = probes["detected_positives"] / probes["positives"]
    if args.human:
        probes = human_probes(probes)
    records = []
    for model, dataset in MODELS.items():
        frame = labels(dataset, human=args.human)
        frozen_layer, adapted_layer = LAYERS[model]
        for split in SPLITS:
            for budget in BUDGETS:
                key = {"model": model, "split": split, "budget": budget}
                records.append({**key, "scorer": "random alarms", "seed": 0, **random_alarms(frame, split, budget)})
                records.append({**key, "scorer": "token-cap rule", "seed": 0, **token_cap(frame, split)})
                for folder, label in baseline_dirs(model, dataset, args.sweep, evaluation):
                    values = from_evaluation(folder, split, budget)
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
    out = OUTPUTS / "analysis" / ("q3_comparison" + ("_sweep" if args.sweep else "") + ("_human" if args.human else "") + ".csv")
    table.reset_index().to_csv(out, index=False)
    print(f"{len(table)} rows -> {out}")


if __name__ == "__main__":
    main()
