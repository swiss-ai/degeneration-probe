"""Materialize the onset position used by degeneration training.

The onset comes from the LLM judge, which reads a rollout and names where it
begins to degenerate by quoting the text. `onset_quotes.py` locates that quote
in the token stream, and this module reads the position it cached.

For every rollout in the dataset, computes:
    - `is_positive`: `stop_reason == "length"` AND a defined onset position
      (see `resolve_onset_position`).
    - `onset_position`: resolved through `resolve_onset_position`, the single
      seam that owns which onset signal to trust. Never read an onset out of
      any other field anywhere in the training pipeline, so that adopting a
      different signal stays a one-line change rather than a pipeline rewrite.
    - `split`: looked up from the existing `splits/*.jsonl` files (prompt-id
      -> split maps, all 10 rollouts of a prompt share one split).
    - `domain`: from `prompts.parquet`.

Negative rollouts (`stop_reason == "eos"`) get `is_positive=False`,
`onset_position=None`. A capped rollout with no resolved onset is excluded from
training rather than treated as either class, and carries the reason it was
excluded so a shrunken positive population can be accounted for.

Output: `paths.onset_labels_path(config)`, one row per rollout, columns per
`ONSET_LABEL_COLUMNS`. This is a standalone parquet -- it does not modify
`labels/<domain>/shard_00000.parquet` or any other existing pipeline output.
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
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "builds" / "degeneration-dataset-apertus-8b-instruct.yaml"

OnsetMetric = Literal["onset_quote"]
DEFAULT_ONSET_METRIC: OnsetMetric = "onset_quote"

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
    # Why a capped rollout carries no onset, so an excluded rollout can be
    # accounted for rather than silently missing.
    "onset_resolution",
]


# --- the swappable onset-position seam -----------------------------------------

def resolve_onset_position(
    row: pd.Series,
    metric: OnsetMetric = DEFAULT_ONSET_METRIC,
    tokenizer=None,
) -> Optional[int]:
    """The one place in the codebase that decides "where does degeneration start"
    for a given rollout. Everything downstream, and by extension every
    probe-training script that reads the materialized table, must go through
    this function rather than reading a raw field directly, so that changing the
    onset-position signal later is a one-line change.

    `row` must carry either `onset_quote_position` (already located, the normal
    path) or an `onset_quote` alongside `generated_token_ids` and a `tokenizer`
    to locate it with.

    Returns None when the metric has no defined onset for this rollout, which
    callers must treat as "exclude from onset-position training" rather than
    filling in a fallback value.
    """
    if metric == "onset_quote":
        # Preferred: the position already located and cached by
        # `onset_quotes.py`, which also records why a quote resolved to nothing.
        cached = row.get("onset_quote_position")
        if cached is not None and not pd.isna(cached):
            return int(cached)
        if "onset_quote_position" in row:
            return None
        quote = row.get("onset_quote")
        if quote is None or (isinstance(quote, float) and pd.isna(quote)) or not quote:
            return None
        if tokenizer is None:
            raise ValueError("resolve_onset_position(metric='onset_quote') requires a tokenizer")
        from degeneration_probe.dataset_gen.onset_quotes import locate_quote

        span = locate_quote(quote, list(row["generated_token_ids"]), tokenizer)
        return None if span is None else int(span.start)

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
    quote_positions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    generations_df = pd.read_parquet(
        paths.generations_shard_path(config, domain, 0),
        columns=["prompt_id", "rollout_idx", "stop_reason", "num_tokens"],
    )
    if quote_positions is None:
        quote_positions = load_quote_positions(config)
    merged = generations_df.merge(
        quote_positions[["prompt_id", "rollout_idx", "onset_quote_position", "resolution"]],
        on=["prompt_id", "rollout_idx"],
        how="left",
    )
    if len(merged) != len(generations_df):
        raise ValueError(
            f"[{domain}] duplicate rows in the quote-position table: "
            f"{len(generations_df)} generations vs {len(merged)} merged rows"
        )

    records: List[dict] = []
    for row in merged.to_dict("records"):
        onset_position = None
        is_positive = False
        resolution = None
        if row["stop_reason"] == "length":
            onset_position = resolve_onset_position(row, metric=onset_metric)
            is_positive = onset_position is not None
            resolution = row.get("resolution") or "unjudged"

        records.append(
            {
                "prompt_id": row["prompt_id"],
                "rollout_idx": int(row["rollout_idx"]),
                "domain": prompt_id_to_domain.get(row["prompt_id"]),
                "split": prompt_id_to_split.get(row["prompt_id"]),
                "stop_reason": row["stop_reason"],
                "num_tokens": int(row["num_tokens"]),
                "is_positive": is_positive,
                "onset_position": onset_position,
                "onset_metric": onset_metric,
                "onset_resolution": resolution,
            }
        )

    return pd.DataFrame.from_records(records, columns=ONSET_LABEL_COLUMNS)


def load_quote_positions(config: DatasetGenConfig) -> pd.DataFrame:
    """The cached quote positions, or a clear instruction to build them."""
    path = paths.onset_quote_positions_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"No quote positions at {path}. The onset comes from the judge's quote located in "
            "the token stream, so build that table first:\n"
            "  python -m degeneration_probe.dataset_gen.onset_quotes --config <build.yaml>"
        )
    return pd.read_parquet(path)


def compute_onset_labels(
    config: DatasetGenConfig,
    domains: Optional[Sequence[str]] = None,
    onset_metric: OnsetMetric = DEFAULT_ONSET_METRIC,
) -> pd.DataFrame:
    domains = domains or sorted(paths.configured_domain_names(config))

    prompts_df = pd.read_parquet(paths.prompts_path(config), columns=["prompt_id", "domain"])
    prompt_id_to_domain = dict(zip(prompts_df["prompt_id"], prompts_df["domain"]))
    prompt_id_to_split = _load_prompt_id_to_split(config)
    quote_positions = load_quote_positions(config)

    frames = [
        compute_onset_labels_for_domain(
            config,
            domain,
            prompt_id_to_domain,
            prompt_id_to_split,
            onset_metric=onset_metric,
            quote_positions=quote_positions,
        )
        for domain in domains
    ]
    return pd.concat(frames, ignore_index=True)


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
        choices=["onset_quote"],
        help="Which onset-position signal to resolve through. Kept as a flag so a future "
        "signal slots in without callers changing.",
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

    excluded = onset_labels_df[
        (onset_labels_df["stop_reason"] == "length") & ~onset_labels_df["is_positive"]
    ]
    if len(excluded):
        print("\nCapped rollouts excluded, by reason:")
        print(excluded["onset_resolution"].value_counts().to_string())
        print("\nExcluded by domain:")
        print(excluded.groupby("domain").size().to_string())


if __name__ == "__main__":
    main()
