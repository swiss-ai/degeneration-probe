"""The early-warning rate a scorer reaches by chance, at a given false-alarm budget.

The reference scorer flags each token independently with probability q and reads
nothing else. Its q is calibrated like any scorer's threshold: the expected
share of healthy validation responses that raise an alarm equals the budget.
A healthy response of T tokens alarms with probability 1 - (1 - q)^T, so q is
found by bisection on the mean of that expression.

Its metrics on a test split then follow in closed form. Post-onset and warning
coverage both equal q. A degenerate response with onset o and length T alarms
with probability 1 - (1 - q)^T, and its first alarm precedes the onset with
probability 1 - (1 - q)^o, so the early-warning rate is the ratio of the sums of
these two over degenerate responses, and the detection rate is the mean of the
first. Degenerate responses run to the token limit while healthy ones stop
early, so even this scorer alarms on most degenerate responses, and the share of
its first alarms before the onset reflects only where onsets fall.

    python scripts/random_alarm_reference.py [--budgets 0.01 0.05 0.10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
MODELS = {"apertus": "apertus-8b-instruct", "llama3p1": "llama3p1-8b-instruct", "mistral": "mistral-7b-instruct-v0p1"}
UNJUDGED = ["judge_failed", "not_found"]


def calibrate(lengths: np.ndarray, budget: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(100):
        q = (low + high) / 2
        if np.mean(1 - (1 - q) ** lengths) > budget:
            high = q
        else:
            low = q
    return low


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--split", default="test_indomain")
    args = parser.parse_args()

    rows = []
    for model, dataset in MODELS.items():
        labels = pd.read_parquet(STORE / f"degeneration-dataset-{dataset}" / "onset_labels" / "onset_labels.parquet")
        labels = labels[~labels["onset_resolution"].isin(UNJUDGED)]
        healthy = labels[(labels["split"] == "val") & ~labels["is_positive"]]["num_tokens"].to_numpy(float)
        test = labels[(labels["split"] == args.split) & labels["is_positive"]]
        onsets, lengths = test["onset_position"].to_numpy(float), test["num_tokens"].to_numpy(float)
        for budget in args.budgets:
            q = calibrate(healthy, budget)
            alarm = 1 - (1 - q) ** lengths
            early = 1 - (1 - q) ** onsets
            rows.append({"model": model, "budget": budget, "q": q, "coverage": q,
                         "detection_rate": alarm.mean(), "early_warning_rate": early.sum() / alarm.sum(),
                         "median_onset_fraction": float(np.median(onsets / lengths)), "degenerate": len(test)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
