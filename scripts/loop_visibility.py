"""How long after the labelled onset a loop becomes visible in the text.

The onset marks the start of the first copy of the repeated text, so the
repetition only shows once the second copy begins. For every degenerate
response this finds the first token at or after the onset that completes an
n-gram already seen since the onset, and reports its distance from the onset
together with the loop period, the distance back to the earlier copy. A model
whose onsets were placed before the loop starts would show long distances.

    python scripts/loop_visibility.py [--split test_indomain] [--n 16]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
MODELS = {"apertus": "apertus-8b-instruct", "llama3p1": "llama3p1-8b-instruct", "mistral": "mistral-7b-instruct-v0p1"}


def first_repeat(ids: list[int], onset: int, n: int) -> tuple[float, float]:
    """Distance from the onset to the first repeated n-gram after it, and the period of that repeat."""
    seen = {}
    for end in range(onset + n, len(ids) + 1):
        key, start = tuple(ids[end - n : end]), end - n
        if key in seen:
            return end - 1 - onset, start - seen[key]
        seen[key] = start
    return float("nan"), float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test_indomain")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--out", type=Path, default=STORE / "outputs" / "analysis" / "loop_visibility.csv")
    args = parser.parse_args()

    rows = []
    for model, dataset in MODELS.items():
        root = STORE / f"degeneration-dataset-{dataset}"
        labels = pd.read_parquet(root / "onset_labels" / "onset_labels.parquet")
        labels = labels[(labels["split"] == args.split) & labels["is_positive"].astype(bool)]
        for domain in sorted(labels["domain"].astype(str).unique()):
            keys = labels[labels["domain"].astype(str) == domain][["prompt_id", "rollout_idx", "onset_position"]]
            for path in sorted((root / "generations" / domain).glob("*.parquet")):
                shard = pd.read_parquet(path, columns=["prompt_id", "rollout_idx", "generated_token_ids"])
                for row in shard.merge(keys, on=["prompt_id", "rollout_idx"]).itertuples(index=False):
                    visible, period = first_repeat(list(map(int, row.generated_token_ids)), int(row.onset_position), args.n)
                    rows.append({"model": model, "domain": domain, "prompt_id": row.prompt_id,
                                 "rollout_idx": int(row.rollout_idx), "onset": int(row.onset_position),
                                 "visible_after": visible, "period": period})
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    summary = table.groupby("model", sort=False)[["visible_after", "period"]].describe(percentiles=[0.25, 0.5, 0.75])
    print(summary.round(0).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
