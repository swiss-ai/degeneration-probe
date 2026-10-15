"""Choose each baseline's hyperparameters on validation, with the probes' rule.

Every setting of the sweep in <outputs>/baselines_sweep/<dataset>/<setting> is
read on every split and budget. Within each baseline family the setting with
the highest validation warning coverage at the 1% budget is selected among
those whose validation post-onset coverage is at least 0.3, the rule that
selects the probes' layer and checkpoint. When no setting clears the floor, the
family falls back to the highest warning coverage overall and is marked so.
The selected setting is then read, unchanged, on the test splits.

    python scripts/select_baseline_sweep.py

Writes <outputs>/analysis/baseline_sweep_all.csv (every setting, split and
budget) and <outputs>/analysis/baseline_sweep_selected.csv (the selected
setting of each family on every split and budget, next to the default one).
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
from scripts.score_baselines import parse_baseline  # noqa: E402

STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
OUTPUTS = STORE / "outputs"
MODELS = {"apertus": "apertus-8b-instruct", "llama3p1": "llama3p1-8b-instruct", "mistral": "mistral-7b-instruct-v0p1"}
SPLITS = ["val", "test_indomain", "test_heldout_domains"]
# The setting each family is reported with so far, for comparison.
DEFAULTS = {"entropy": "entropy_w128", "neglogp": "neglogp_w128", "repetition_trailing": "repetition_trailing_n2_w256",
            "rep_l": "rep_l_l128_w128", "ngram": {"apertus": "ngram_30", "llama3p1": "ngram_8", "mistral": "ngram_16"}}
SELECTION_BUDGET = 0.01
POST_FLOOR = 0.3
METRICS = ["negative_fpr", "detection_rate", "rollout_auc", "in_pattern_recall", "warning_recall_256",
           "preonset_coverage_256", "fired_before_frontier", "median_offset"]


def read_all() -> pd.DataFrame:
    rows = []
    for model, dataset in MODELS.items():
        root = OUTPUTS / "baselines_sweep" / dataset
        for setting in sorted(p.name for p in root.iterdir() if (p / "evaluation").is_dir()):
            family, params = parse_baseline(setting)
            for split in SPLITS:
                frame = read_depth(root / setting / "evaluation", split)
                if frame is None:
                    continue
                frame = frame.assign(model=model, family=family, setting=setting,
                                     params=" ".join(f"{k}={v}" for k, v in params.items()))
                frame["detection_rate"] = frame["detected_positives"] / frame["positives"]
                rows.append(frame)
    frame = pd.concat(rows, ignore_index=True)
    frame = frame.rename(columns={"target_negative_fpr": "budget"})
    return frame[["model", "family", "setting", "params", "split", "budget"] + METRICS]


def select(frame: pd.DataFrame) -> pd.DataFrame:
    val = frame[(frame["split"] == "val") & np.isclose(frame["budget"], SELECTION_BUDGET)]
    chosen = []
    for (model, family), group in val.groupby(["model", "family"]):
        eligible = group[group["in_pattern_recall"] >= POST_FLOOR]
        pool = eligible if len(eligible) else group
        best = pool.sort_values(["warning_recall_256", "in_pattern_recall"], ascending=False).iloc[0]
        default = DEFAULTS[family][model] if family == "ngram" else DEFAULTS[family]
        chosen.append({"model": model, "family": family, "selected": best["setting"],
                       "default": default, "floor_met": bool(len(eligible)),
                       "settings": len(group), "eligible": len(eligible)})
    return pd.DataFrame(chosen)


def main() -> None:
    frame = read_all()
    out = OUTPUTS / "analysis"
    frame.to_csv(out / "baseline_sweep_all.csv", index=False)
    chosen = select(frame)
    print(chosen.to_string(index=False))

    rows = []
    for row in chosen.itertuples(index=False):
        for role, setting in (("selected", row.selected), ("default", row.default)):
            part = frame[(frame["model"] == row.model) & (frame["setting"] == setting)]
            rows.append(part.assign(role=role, floor_met=row.floor_met))
    selected = pd.concat(rows, ignore_index=True)
    selected.to_csv(out / "baseline_sweep_selected.csv", index=False)

    view = selected[(selected["split"] == "test_indomain") & np.isclose(selected["budget"], 0.05)]
    print("\n== test_indomain, budget 5% ==")
    print(view[["model", "family", "role", "setting", "detection_rate", "in_pattern_recall",
                "warning_recall_256", "preonset_coverage_256", "fired_before_frontier", "median_offset"]].to_string(index=False))
    print(f"\n{len(frame)} rows -> {out / 'baseline_sweep_all.csv'}")
    print(f"{len(selected)} rows -> {out / 'baseline_sweep_selected.csv'}")


if __name__ == "__main__":
    main()
