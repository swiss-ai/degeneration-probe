"""What selecting a probe on a response-level metric costs in warning coverage.

Each candidate is one recipe read at one depth, averaged over its seeds. A
practitioner picks the candidate that scores best on validation under some
metric, and deploys it. This reports, for each selection metric, the test
warning coverage of the candidate it picks, and the ratio to the candidate
picked by validation warning coverage itself.

Three selection axes are reported, because the cost is not the same along all
of them:

    recipe   every Apertus recipe read at one layer (frozen only, and with
             the adapted recipe added)
    depth    within each recipe, every depth; the median and worst ratio over
             recipes
    pool     every (recipe, depth) candidate together

Only depths that every seed can select (post-onset coverage floor) are
candidates, so a metric is never credited for a probe the selection rule would
refuse.

    python scripts/selection_regret.py [--budget 0.05] [--layer 15]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/outputs/analysis")
OUTPUTS = ANALYSIS.parent
METRICS = {"AUROC": "rollout_auc", "best accuracy": "best_accuracy", "Warn256": "warning_recall_256", "Post": "in_pattern_recall"}
TARGET = "warning_recall_256"


def candidates(model: str, budget: float) -> pd.DataFrame:
    table = pd.read_csv(ANALYSIS / "selected_metrics.csv")
    table = table[table["model"] == model]
    pool = ANALYSIS / "response_metrics_pool.csv"
    if pool.is_file():
        table["scorer_dir"] = [f"{OUTPUTS}/{r}/{a}/selected/layer_{int(l):02d}" for r, a, l in
                               zip(table["run_name"], table["attempt"], table["layer"])]
        acc = pd.read_csv(pool)[["scorer_dir", "split", "best_accuracy"]]
        table = table.merge(acc, on=["scorer_dir", "split"], how="left")
    else:
        table["best_accuracy"] = np.nan
    selectable = table[table["split"] == "val"].groupby(["recipe", "layer"])["selectable"].all()
    rows = table[table["target_negative_fpr"] == budget]
    means = rows.groupby(["recipe", "layer", "split"])[list(METRICS.values())].mean().unstack("split")
    means.columns = [f"{m}_{s}" for m, s in means.columns]
    means["selectable"] = selectable
    return means[means["selectable"]].reset_index()


def pick(frame: pd.DataFrame) -> dict:
    """Test warning coverage of the candidate each metric picks on validation."""
    out = {}
    for name, column in METRICS.items():
        valid = frame.dropna(subset=[f"{column}_val"])
        if valid.empty:
            out[name] = np.nan
            continue
        best = valid.loc[valid[f"{column}_val"].idxmax()]
        out[name] = best[f"{TARGET}_test_indomain"]
        out[name + " pick"] = f"{best['recipe']} L{int(best['layer'])}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="apertus")
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()

    frame = candidates(args.model, args.budget)
    adapted = frame["recipe"].str.endswith("+LoRA")
    results = {
        f"recipe axis, frozen, L{args.layer}": pick(frame[~adapted & (frame["layer"] == args.layer)]),
        f"recipe axis, with adapted, L{args.layer}": pick(frame[frame["layer"] == args.layer]),
        "pool, frozen": pick(frame[~adapted]),
        "pool, with adapted": pick(frame),
    }
    print(f"model {args.model}, budget {args.budget:.0%}, test in-domain Warn256 of the candidate each metric picks on validation\n")
    for axis, picked in results.items():
        line = ", ".join(f"{name} {picked[name]:.3f} ({picked.get(name + ' pick', '')})" for name in METRICS)
        ratio = picked["Warn256"] / picked["AUROC"] if picked["AUROC"] else np.nan
        print(f"{axis}: {line}\n    Warn-selected / AUROC-selected = {ratio:.2f}x")

    ratios = []
    for recipe, group in frame[~adapted].groupby("recipe"):
        if len(group) < 2:
            continue
        picked = pick(group)
        ratios.append(picked["Warn256"] / picked["AUROC"])
    ratios = pd.Series(ratios)
    print(f"\ndepth axis, {len(ratios)} frozen recipes: Warn-selected / AUROC-selected median {ratios.median():.2f}x, "
          f"worst {ratios.max():.2f}x, AUROC optimal in {(ratios <= 1.0).sum()} of {len(ratios)}")


if __name__ == "__main__":
    main()
