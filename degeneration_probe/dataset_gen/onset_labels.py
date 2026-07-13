"""Onset-label materialization for the degeneration-onset probes (`probe_N` /
multi-horizon).

For every rollout in the dataset, computes:
    - `is_positive`: `stop_reason == "length"` AND a defined onset position
      under the active onset metric (see `resolve_onset_position`). This is
      the settled positive/negative rollout definition -- validated this
      session against independent LLM-judge ground truth (stop_reason ==
      "length" is 99.6% precise for real degeneration; `repetition_score >
      0.8` alone is not, and must never be used as the primary label -- see
      the project's onset-probes implementation prompt for the full
      validation).
    - `onset_position`: resolved through `resolve_onset_position`, a single
      swappable seam over which onset-position signal to trust. Do not read
      `lrs_first_start_normalized_growing` (or any other field) directly
      anywhere else in the probe-training pipeline -- always go through this
      function, so that adopting a better signal later (e.g. the LLM judge's
      `onset_quote`, once populated at scale) is a one-line config change,
      not a pipeline rewrite.
    - `split`: looked up from the existing `splits/*.jsonl` files (prompt-id
      -> split maps, all 10 rollouts of a prompt share one split).
    - `domain`: from `prompts.parquet`.

Negative rollouts (`stop_reason == "eos"`) get `is_positive=False`,
`onset_position=None`, regardless of any heuristic flag (repetition_score,
LRS score, etc.) -- see the module-level docstring in `label.py` and the
project's implementation prompt for why those heuristics are not trustworthy
enough to use as a corpus-wide positive-label source.

Output: `paths.onset_labels_path(config)`, one row per rollout, columns per
`ONSET_LABEL_COLUMNS`. This is a new, standalone parquet -- it does not
modify `labels/<domain>/shard_00000.parquet` or any other existing pipeline
output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import pandas as pd

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.dataset_gen.label import write_shard_atomic

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"

# Candidate horizons for the sparsity sanity check / eventual probe_N training.
DEFAULT_HORIZONS = (5, 20, 50, 100, 200)

OnsetMetric = Literal["lrs_normalized_growing", "onset_quote"]
DEFAULT_ONSET_METRIC: OnsetMetric = "lrs_normalized_growing"

ONSET_LABEL_COLUMNS = [
    "prompt_id",
    "rollout_idx",
    "domain",
    "split",
    "stop_reason",
    "num_tokens",
    "is_positive",
    "onset_position",
    "onset_metric",
    "n_lrs_regions_normalized_growing",
]


# --- the swappable onset-position seam -----------------------------------------

def resolve_onset_position(
    row: pd.Series,
    metric: OnsetMetric = DEFAULT_ONSET_METRIC,
    tokenizer=None,
) -> Optional[int]:
    """The one place in the codebase that decides "where does degeneration start"
    for a given rollout. Everything downstream (subtask A's materialized table,
    and by extension every probe-training script that reads it) must go through
    this function rather than reading an LRS/judge field directly, so that
    changing the onset-position signal later is a one-line config change.

    `row` must contain the fields the chosen `metric` needs:
      - "lrs_normalized_growing" (current default): `lrs_length_normalized_growing`,
        `lrs_first_start_normalized_growing` (from `labels/<domain>/shard_00000.parquet`).
      - "onset_quote": `onset_quote` (from `llm_judge/results_*.parquet`, once
        populated -- see the module docstring in `llm_judge.py`) plus
        `generated_token_ids` (from `generations/<domain>/shard_00000.parquet`);
        requires `tokenizer` to locate the quote's token span.

    Returns None if the metric has no defined onset for this rollout (e.g. no
    LRS match at all under `min_length`, or no `onset_quote` recorded) --
    callers must treat that as "exclude from onset-position training", never
    invent a fallback value.
    """
    if metric == "lrs_normalized_growing":
        length = row.get("lrs_length_normalized_growing")
        if length is None or pd.isna(length) or int(length) == 0:
            return None
        return int(row["lrs_first_start_normalized_growing"])

    if metric == "onset_quote":
        quote = row.get("onset_quote")
        if quote is None or (isinstance(quote, float) and pd.isna(quote)) or not quote:
            return None
        if tokenizer is None:
            raise ValueError("resolve_onset_position(metric='onset_quote') requires a tokenizer")
        from degeneration_probe.utils.tokenization import find_string_in_tokens
        import torch

        token_ids = torch.as_tensor(list(row["generated_token_ids"]))
        try:
            span = find_string_in_tokens(quote, token_ids, tokenizer)
        except (AssertionError, ValueError):
            return None
        return int(span.start)

    raise ValueError(f"Unknown onset_metric {metric!r}; expected one of the OnsetMetric literal values")


# --- split lookup ----------------------------------------------------------------

def _load_prompt_id_to_split(config: DatasetGenConfig) -> Dict[str, str]:
    """Read every `splits/*.jsonl` file and invert to a prompt_id -> split_name map.

    Doesn't hardcode the set of split names -- whatever `.jsonl` files exist
    under `paths.splits_dir` are used, so a future split (e.g. subtask L's
    LoRA-pilot in-domain held-out stand-in) is picked up automatically.
    """
    prompt_id_to_split: Dict[str, str] = {}
    splits_dir = paths.splits_dir(config)
    for split_file in sorted(splits_dir.glob("*.jsonl")):
        split_name = split_file.stem
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                prompt_id = json.loads(line)["prompt_id"]
                prompt_id_to_split[prompt_id] = split_name
    return prompt_id_to_split


# --- per-domain / corpus-wide computation -----------------------------------------

def compute_onset_labels_for_domain(
    config: DatasetGenConfig,
    domain: str,
    prompt_id_to_domain: Dict[str, str],
    prompt_id_to_split: Dict[str, str],
    onset_metric: OnsetMetric = DEFAULT_ONSET_METRIC,
) -> pd.DataFrame:
    generations_df = pd.read_parquet(
        paths.generations_shard_path(config, domain, 0),
        columns=["prompt_id", "rollout_idx", "stop_reason", "num_tokens"],
    )
    labels_df = pd.read_parquet(
        paths.labels_shard_path(config, domain, 0),
        columns=[
            "prompt_id",
            "rollout_idx",
            "lrs_length_normalized_growing",
            "lrs_first_start_normalized_growing",
            "lrs_region_starts_normalized_growing",
        ],
    )
    merged = generations_df.merge(labels_df, on=["prompt_id", "rollout_idx"], how="inner")
    if len(merged) != len(generations_df):
        raise ValueError(
            f"[{domain}] generations/labels row mismatch after merge: "
            f"{len(generations_df)} generations vs {len(merged)} merged rows"
        )

    records: List[dict] = []
    for row in merged.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        onset_position = None
        is_positive = False
        if row.stop_reason == "length":
            onset_position = resolve_onset_position(row_series, metric=onset_metric)
            is_positive = onset_position is not None

        region_starts = row.lrs_region_starts_normalized_growing
        n_regions = len(region_starts) if region_starts is not None else 0

        records.append(
            {
                "prompt_id": row.prompt_id,
                "rollout_idx": int(row.rollout_idx),
                "domain": prompt_id_to_domain.get(row.prompt_id),
                "split": prompt_id_to_split.get(row.prompt_id),
                "stop_reason": row.stop_reason,
                "num_tokens": int(row.num_tokens),
                "is_positive": is_positive,
                "onset_position": onset_position,
                "onset_metric": onset_metric,
                "n_lrs_regions_normalized_growing": n_regions,
            }
        )

    return pd.DataFrame.from_records(records, columns=ONSET_LABEL_COLUMNS)


def compute_onset_labels(
    config: DatasetGenConfig,
    domains: Optional[Sequence[str]] = None,
    onset_metric: OnsetMetric = DEFAULT_ONSET_METRIC,
) -> pd.DataFrame:
    domains = domains or sorted(paths.configured_domain_names(config))

    prompts_df = pd.read_parquet(paths.prompts_path(config), columns=["prompt_id", "domain"])
    prompt_id_to_domain = dict(zip(prompts_df["prompt_id"], prompts_df["domain"]))
    prompt_id_to_split = _load_prompt_id_to_split(config)

    frames = [
        compute_onset_labels_for_domain(
            config, domain, prompt_id_to_domain, prompt_id_to_split, onset_metric=onset_metric
        )
        for domain in domains
    ]
    return pd.concat(frames, ignore_index=True)


# --- sanity checks: eligible-token counts per horizon, multi-region flag ----------

def eligible_token_counts_per_horizon(
    onset_labels_df: pd.DataFrame, horizons: Sequence[int] = DEFAULT_HORIZONS
) -> pd.DataFrame:
    """Reproduce the label-sparsity table from the implementation prompt: for
    each horizon N, total eligible tokens and how many of them are positive
    under `y_N(t) = 1 if (onset - t) <= N else 0`.

    Eligible tokens = every position in negative rollouts (all label 0) plus
    positions `t in [0, onset]` (onset inclusive) in positive rollouts with a
    defined onset. For a positive rollout, the number of positions with
    `(onset - t) <= N` is `min(N, onset) + 1` (all of them if `onset <= N`,
    otherwise exactly `N + 1`).

    Negative rollout here means `stop_reason == "eos"` specifically (not just
    "not positive") -- rollouts that hit the token cap (`stop_reason ==
    "length"`) but have no defined onset under the active metric are neither
    positive nor negative and must be excluded from both sums entirely, per
    the onset-position metric's documented limitations.
    """
    negative = onset_labels_df[onset_labels_df["stop_reason"] == "eos"]
    positive = onset_labels_df[onset_labels_df["is_positive"]]

    n_negative_tokens = int(negative["num_tokens"].sum())
    onsets = positive["onset_position"].astype(int).to_numpy()
    n_positive_eligible_tokens = int((onsets + 1).sum())
    total_eligible = n_negative_tokens + n_positive_eligible_tokens

    rows = []
    for horizon in horizons:
        positive_tokens = int(sum(min(horizon, onset) + 1 for onset in onsets))
        rows.append(
            {
                "horizon": horizon,
                "eligible_tokens": total_eligible,
                "positive_tokens": positive_tokens,
                "positive_rate": positive_tokens / total_eligible if total_eligible else float("nan"),
            }
        )
    return pd.DataFrame.from_records(rows)


# --- CLI entry point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="Path to a DatasetGenConfig YAML file (default: degeneration-dataset-apertus-8b-instruct.yaml).",
    )
    parser.add_argument(
        "--domains", nargs="*", default=None,
        help="Subset of domains to process (default: all configured domains).",
    )
    parser.add_argument(
        "--onset-metric", type=str, default=DEFAULT_ONSET_METRIC,
        choices=["lrs_normalized_growing", "onset_quote"],
        help="Which onset-position signal to resolve through (default: lrs_normalized_growing). "
        "'onset_quote' is not usable yet -- see llm_judge.py, subtask L in the implementation prompt.",
    )
    parser.add_argument(
        "--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS),
        help="Horizons (in tokens) to report eligible/positive token counts for, as a sanity check "
        "against the label-sparsity table (default: 5 20 50 100 200).",
    )
    args = parser.parse_args()

    config = DatasetGenConfig.from_yaml(args.config)

    onset_labels_df = compute_onset_labels(config, domains=args.domains, onset_metric=args.onset_metric)

    out_path = paths.onset_labels_path(config)
    write_shard_atomic(onset_labels_df, out_path)
    print(f"Wrote {len(onset_labels_df)} rollout rows to {out_path}")

    n_positive = int(onset_labels_df["is_positive"].sum())
    n_length = int((onset_labels_df["stop_reason"] == "length").sum())
    n_no_onset = n_length - n_positive
    print(
        f"is_positive: {n_positive} / {len(onset_labels_df)} rollouts "
        f"({n_length} truncated total, {n_no_onset} of those have no defined onset under "
        f"onset_metric={args.onset_metric!r} and are excluded)"
    )

    by_split = onset_labels_df[onset_labels_df["is_positive"]].groupby("split").size()
    print("Positive rollouts by split:")
    print(by_split.to_string())

    multi_region = onset_labels_df[
        onset_labels_df["is_positive"] & (onset_labels_df["n_lrs_regions_normalized_growing"] > 1)
    ]
    print(
        f"\n[multi-episode sanity check] {len(multi_region)} / {n_positive} positive rollouts have "
        f">1 disjoint LRS region under the normalized-growing metric -- these are candidates worth "
        f"a manual spot-check for a second, distinct repeat episode this metric's single "
        f"`onset_position` might not represent (see subtask A's multi-episode check in the "
        f"implementation prompt). Not automatically resolved here."
    )
    if len(multi_region) > 0:
        sample_cols = ["prompt_id", "rollout_idx", "n_lrs_regions_normalized_growing"]
        print(multi_region[sample_cols].head(10).to_string(index=False))

    print("\nEligible/positive token counts per horizon (sanity check vs. the label-sparsity table):")
    counts_df = eligible_token_counts_per_horizon(onset_labels_df, horizons=args.horizons)
    print(counts_df.to_string(index=False))


if __name__ == "__main__":
    main()
