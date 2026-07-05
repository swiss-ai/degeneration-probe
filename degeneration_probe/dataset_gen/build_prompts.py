"""Build the dataset-v2 pilot prompt pool and train/val/test splits.

For every source configured in ``config.in_domain_sources`` /
``config.held_out_sources`` this script:

1. Loads the real HF dataset (``datasets.load_dataset``).
2. Extracts the raw prompt text per row, auto-detecting whether the
   configured ``prompt_field`` holds a flat string or a conversation-list
   (``[{"role": ..., "content": ...}, ...]``) payload -- mirroring the
   ``_first_present``-style auto-detection in
   ``degeneration_probe/data/converters.py``.
3. Shuffles deterministically (``config.seed``) and samples ``n_prompts``
   rows (clamped to however many rows the source actually has).
4. Assigns a stable ``prompt_id`` (``f"{domain}_{index:05d}"``) and the
   ``domain`` / ``held_out_domain`` fields.

The combined prompt pool is written to ``paths.prompts_path(config)``. It is
then split into ``train`` / ``val`` / ``test_indomain`` (in-domain sources
only, per ``config.split_fractions``) and ``test_heldout_domains`` (all
held-out-domain prompts, entirely reserved for OOD eval), written to
``paths.split_path(config, name)``.

Some configured sources are far too large to fully download for a small
prompt sample (e.g. ``llama_nemotron``'s "code" split is ~48GB across two
JSONL files). For any HF repo whose total file size exceeds
``LARGE_REPO_STREAMING_THRESHOLD_BYTES`` we use ``datasets`` streaming mode
and only read a bounded prefix of the stream
(``STREAMING_PREFIX_SCAN_SIZE`` examples), then apply the exact same
deterministic Fisher-Yates shuffle-and-sample (``sample_source_rows``) used
for fully-downloaded sources, over just that prefix. This is a deliberate
tradeoff: the sample is drawn from the first ``STREAMING_PREFIX_SCAN_SIZE``
rows of the stream (in whatever order the HF data loader concatenates the
underlying shard files), not a uniform sample over the entire multi-GB/
multi-million-row split. This keeps the bounded work independent of the
split's total size (no full-file download, no unbounded shuffle-buffer fill)
at the cost of that prefix bias. ``STREAMING_PREFIX_SCAN_SIZE`` should be
raised whenever ``n_prompts`` for a streamed source grows, to keep the
sampled fraction of the prefix (and thus its diversity) reasonable -- it was
5,000 for the 70-prompt pilot (1.4% sampled), raised to 20,000 for the
~600-prompt full-scale build (3% sampled).
Small sources are fully downloaded and sampled with an exact Fisher-Yates
shuffle over all rows.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import PilotDatasetConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset_gen" / "pilot_v1.yaml"

# HF repos whose total (all-files) size exceeds this many bytes are read in
# streaming mode with a bounded prefix scan instead of being fully
# downloaded. 5 GiB comfortably covers every configured source except
# llama_nemotron (whose "code" split alone is ~48GB).
LARGE_REPO_STREAMING_THRESHOLD_BYTES = 5 * 1024**3

# Number of examples read from the front of the stream for large (streamed)
# sources, before applying the same deterministic shuffle-and-sample used for
# fully-downloaded sources. Bounds the work to a fixed prefix scan regardless
# of the split's actual size (e.g. llama_nemotron's "code" split spans two
# 18-48GB JSONL files -- we only ever read this many rows from the front of
# them). See module docstring for the tradeoff this implies.
STREAMING_PREFIX_SCAN_SIZE = 20_000

SourceRow = Tuple[Any, str]  # (source_row_id, prompt_text)


# --- prompt-field extraction -------------------------------------------------

def extract_prompt_text(value: Any) -> str:
    """Extract prompt text from a single HF dataset cell.

    Handles both flat-string prompt columns (e.g. DeepMath's "question") and
    conversation-list schemas (e.g. IF_sft_data_verified's "messages" or
    llama_nemotron's "input", each a list of ``{"role": ..., "content": ...}``
    dicts) -- the same style of auto-detection used by
    ``degeneration_probe/data/converters.py``'s ``_first_present`` helper.
    """
    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        # Prefer the first user turn.
        for turn in value:
            if isinstance(turn, dict) and turn.get("role") == "user" and turn.get("content"):
                return str(turn["content"])
        # Fall back to the first turn with any content at all.
        for turn in value:
            if isinstance(turn, dict) and turn.get("content"):
                return str(turn["content"])
        raise ValueError(f"Could not extract prompt text from conversation-list value: {value!r}")

    if value is None:
        raise ValueError("Prompt field value is None; cannot extract prompt text")

    return str(value)


# --- sampling (network-free, unit-testable) ----------------------------------

def sample_source_rows(rows: Sequence[SourceRow], n_prompts: int, seed: int) -> List[SourceRow]:
    """Deterministically shuffle ``rows`` and take the first ``n_prompts``.

    Clamps to ``len(rows)`` (with a warning) if the source has fewer rows
    than requested, rather than raising.
    """
    n = min(n_prompts, len(rows))
    if n < n_prompts:
        print(
            f"WARNING: requested n_prompts={n_prompts} but only {len(rows)} rows "
            f"are available; sampling all {n}."
        )
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i] for i in order[:n]]


def build_domain_prompts(config: PilotDatasetConfig, source: Dict[str, Any], sampled_rows: Sequence[SourceRow]) -> pd.DataFrame:
    """Turn a sampled (source_row_id, prompt_text) list into the prompts-table rows for one domain."""
    domain = source["name"]
    held_out = paths.is_held_out_domain(config, domain)

    records = []
    for i, (source_row_id, prompt_text) in enumerate(sampled_rows):
        records.append(
            {
                "prompt_id": f"{domain}_{i:05d}",
                "domain": domain,
                "source_dataset": source["hf_repo"],
                "source_row_id": source_row_id,
                "prompt_text": prompt_text,
                "held_out_domain": held_out,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["prompt_id", "domain", "source_dataset", "source_row_id", "prompt_text", "held_out_domain"],
    )


# --- HF loading (network) -----------------------------------------------------

def _is_large_repo(hf_repo: str) -> bool:
    """Whether ``hf_repo``'s total file size exceeds the streaming threshold."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        info = api.dataset_info(hf_repo, files_metadata=True)
    except Exception as exc:  # pragma: no cover - network-dependent
        print(f"WARNING: could not fetch repo size for {hf_repo!r} ({exc}); assuming small (full download).")
        return False

    total_bytes = sum((sibling.size or 0) for sibling in info.siblings)
    return total_bytes > LARGE_REPO_STREAMING_THRESHOLD_BYTES


def _fetch_full(hf_repo: str, hf_subset: Any, hf_split: str, prompt_field: str) -> List[SourceRow]:
    from datasets import load_dataset

    ds = load_dataset(hf_repo, hf_subset, split=hf_split)
    column_values = ds[prompt_field]
    return [(i, extract_prompt_text(v)) for i, v in enumerate(column_values)]


def _fetch_streaming_sample(
    hf_repo: str, hf_subset: Any, hf_split: str, prompt_field: str, n_prompts: int, seed: int
) -> List[SourceRow]:
    """Read a bounded prefix of a streamed split and sample from it.

    Only ever reads ``STREAMING_PREFIX_SCAN_SIZE`` examples from the front of
    the stream (regardless of the split's total size), then reuses the same
    deterministic ``sample_source_rows`` shuffle-and-sample used for
    fully-downloaded sources. See module docstring for the tradeoff.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_repo, hf_subset, split=hf_split, streaming=True)
    ds = ds.take(STREAMING_PREFIX_SCAN_SIZE)

    prefix_rows: List[SourceRow] = [
        (idx, extract_prompt_text(example[prompt_field])) for idx, example in enumerate(ds)
    ]

    if len(prefix_rows) < STREAMING_PREFIX_SCAN_SIZE:
        print(
            f"NOTE: streaming source {hf_repo!r} split={hf_split!r} has only "
            f"{len(prefix_rows)} rows total (< prefix scan size "
            f"{STREAMING_PREFIX_SCAN_SIZE}); scanned all of them."
        )

    return sample_source_rows(prefix_rows, n_prompts, seed)


def fetch_and_sample_source(source: Dict[str, Any], seed: int) -> List[SourceRow]:
    """Load ``source``'s HF dataset and return its sampled (row_id, prompt_text) rows."""
    hf_repo = source["hf_repo"]
    hf_subset = source.get("hf_subset")
    hf_split = source["hf_split"]
    prompt_field = source["prompt_field"]
    n_prompts = source["n_prompts"]

    if _is_large_repo(hf_repo):
        print(f"  (repo exceeds {LARGE_REPO_STREAMING_THRESHOLD_BYTES / 1024**3:.0f} GiB -- using streaming sample)")
        return _fetch_streaming_sample(hf_repo, hf_subset, hf_split, prompt_field, n_prompts, seed)

    rows = _fetch_full(hf_repo, hf_subset, hf_split, prompt_field)
    return sample_source_rows(rows, n_prompts, seed)


# --- assembling the full prompt pool ------------------------------------------

def build_prompts_table(config: PilotDatasetConfig) -> pd.DataFrame:
    frames = []
    for source in [*config.in_domain_sources, *config.held_out_sources]:
        print(
            f"Loading source {source['name']!r} ({source['hf_repo']}, "
            f"subset={source.get('hf_subset')!r}, split={source['hf_split']!r})..."
        )
        sampled_rows = fetch_and_sample_source(source, config.seed)
        domain_df = build_domain_prompts(config, source, sampled_rows)
        print(f"  -> sampled {len(domain_df)} prompts for domain {source['name']!r}")
        frames.append(domain_df)

    return pd.concat(frames, ignore_index=True)


# --- splitting (network-free, unit-testable) ---------------------------------

def assign_splits(config: PilotDatasetConfig, prompts_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Assign every prompt_id to exactly one of train/val/test_indomain/test_heldout_domains."""
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test_indomain": [], "test_heldout_domains": []}

    in_domain_names = sorted({source["name"] for source in config.in_domain_sources})

    for domain in in_domain_names:
        ids = prompts_df.loc[prompts_df["domain"] == domain, "prompt_id"].tolist()

        order = list(range(len(ids)))
        random.Random(config.seed).shuffle(order)
        shuffled_ids = [ids[i] for i in order]

        n = len(shuffled_ids)
        n_train = min(int(round(n * config.split_fractions["train"])), n)
        n_val = min(int(round(n * config.split_fractions["val"])), n - n_train)

        splits["train"].extend(shuffled_ids[:n_train])
        splits["val"].extend(shuffled_ids[n_train : n_train + n_val])
        splits["test_indomain"].extend(shuffled_ids[n_train + n_val :])

    splits["test_heldout_domains"] = prompts_df.loc[prompts_df["held_out_domain"], "prompt_id"].tolist()

    return splits


def sanity_check(config: PilotDatasetConfig, prompts_df: pd.DataFrame, splits: Dict[str, List[str]]) -> None:
    """Print per-domain/per-split counts and raise if any disjointness invariant is violated."""
    print("\nPer-domain prompt counts:")
    domain_counts = prompts_df.groupby("domain").size()
    for domain in sorted(paths.configured_domain_names(config)):
        count = int(domain_counts.get(domain, 0))
        kind = "held_out" if paths.is_held_out_domain(config, domain) else "in_domain"
        print(f"  {domain:25s} ({kind:9s}): {count}")

    print("\nPer-split prompt counts:")
    for split_name in ("train", "val", "test_indomain", "test_heldout_domains"):
        print(f"  {split_name:20s}: {len(splits[split_name])}")

    split_sets = {name: set(ids) for name, ids in splits.items()}

    # No duplicate prompt_ids within a single split.
    for name, ids in splits.items():
        if len(ids) != len(split_sets[name]):
            raise AssertionError(f"Split {name!r} contains duplicate prompt_id values")

    # Pairwise disjointness across all four splits.
    names = list(split_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_sets[names[i]] & split_sets[names[j]]
            if overlap:
                raise AssertionError(
                    f"Splits {names[i]!r} and {names[j]!r} are not disjoint: "
                    f"{len(overlap)} overlapping prompt_id(s), e.g. {sorted(overlap)[:5]}"
                )

    # Held-out-domain prompt_ids must live in test_heldout_domains and nowhere else.
    held_out_ids = set(prompts_df.loc[prompts_df["held_out_domain"], "prompt_id"])
    non_heldout_ids = split_sets["train"] | split_sets["val"] | split_sets["test_indomain"]
    leaked = held_out_ids & non_heldout_ids
    if leaked:
        raise AssertionError(
            f"{len(leaked)} held-out-domain prompt_id(s) leaked into train/val/test_indomain: "
            f"{sorted(leaked)[:5]}"
        )
    missing = held_out_ids - split_sets["test_heldout_domains"]
    if missing:
        raise AssertionError(
            f"{len(missing)} held-out-domain prompt_id(s) are missing from test_heldout_domains: "
            f"{sorted(missing)[:5]}"
        )

    # And test_heldout_domains must contain *only* held-out-domain prompt_ids.
    non_held_out_ids = set(prompts_df.loc[~prompts_df["held_out_domain"], "prompt_id"])
    stray = split_sets["test_heldout_domains"] & non_held_out_ids
    if stray:
        raise AssertionError(
            f"{len(stray)} in-domain prompt_id(s) found in test_heldout_domains: {sorted(stray)[:5]}"
        )

    print("\nSanity checks passed: all 4 splits are pairwise disjoint in prompt_id, and every "
          "held-out-domain prompt_id is in test_heldout_domains only.")


# --- writing outputs -----------------------------------------------------------

def write_prompts_parquet(config: PilotDatasetConfig, prompts_df: pd.DataFrame) -> Path:
    out_path = paths.prompts_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_df.to_parquet(out_path, index=False)
    return out_path


def write_splits_jsonl(config: PilotDatasetConfig, splits: Dict[str, List[str]]) -> Dict[str, Path]:
    written: Dict[str, Path] = {}
    for name, ids in splits.items():
        out_path = paths.split_path(config, name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for prompt_id in ids:
                f.write(json.dumps({"prompt_id": prompt_id}) + "\n")
        written[name] = out_path
    return written


# --- entry point ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a PilotDatasetConfig YAML file (default: configs/dataset_gen/pilot_v1.yaml).",
    )
    args = parser.parse_args()

    config = PilotDatasetConfig.from_yaml(args.config)

    prompts_df = build_prompts_table(config)
    splits = assign_splits(config, prompts_df)
    sanity_check(config, prompts_df, splits)

    prompts_out_path = write_prompts_parquet(config, prompts_df)
    split_out_paths = write_splits_jsonl(config, splits)

    print(f"\nWrote {len(prompts_df)} prompts to {prompts_out_path}")
    for name, out_path in split_out_paths.items():
        print(f"Wrote split {name!r} ({len(splits[name])} prompt_ids) to {out_path}")


if __name__ == "__main__":
    main()
