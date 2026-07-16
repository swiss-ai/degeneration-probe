"""Export a manual-inspection package for LLM-judge calibration verdicts.

Builds two files meant to be handed out together to lab-mates for manual
review:

    - ``judge_review_data.json``: every judge verdict worth double-checking
      by hand, plus a per-row ``assigned_to`` reviewer name.
    - ``judge_review.html`` (copied as-is, unmodified, from this repo):
      the offline review app that reads the JSON above.

Scope: only rollouts that hit the token cap without reaching EOS
(``stop_reason == "length"``, i.e. Section 7's "truncated" stratum in
``llm_judge.select_calibration_sample``) are included. That cap is a
necessary (not sufficient) condition for real degeneration -- see
``onset_labels.py``'s module docstring, which validated ``stop_reason ==
"length"`` at 99.6% precision against LLM-judge ground truth -- so this is
exactly the population where a manual pass is worth the effort: checking the
judge's ``onset_quote`` position, and catching the rare misclassification
(most often a false negative -- the judge reads genuine progress, e.g.
brute-force enumeration, into what is actually a non-progressing loop).

Only rows already judged (``status == "ok"``) by the chosen backend are
exported -- a failed/unjudged row has no verdict to review yet.

Usage:
    .venv/bin/python scripts/manual-inspection/export_judge_review.py \\
        --config configs/dataset/degeneration-dataset-apertus-8b-instruct.yaml \\
        --reviewers alice,bob,carlo \\
        --out-dir /path/to/handoff/dir
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"
HTML_APP_PATH = Path(__file__).resolve().parent / "judge_review.html"

# Confidence tiers the judge itself reports, worst-first -- used only to bias
# *which* rows get included under --max-samples, not to filter anything out.
_CONFIDENCE_PRIORITY = {"low": 0, "medium": 1, "high": 2, None: 1}


def _load_labels_for_heuristics(config: DatasetGenConfig, domains: List[str]) -> pd.DataFrame:
    frames = []
    for domain in domains:
        p = paths.labels_shard_path(config, domain, 0)
        if not p.exists():
            continue
        frames.append(
            pd.read_parquet(
                p, columns=["prompt_id", "rollout_idx", "repetition_score", "lrs_period_repeat_count"]
            )
        )
    if not frames:
        return pd.DataFrame(columns=["prompt_id", "rollout_idx", "max_repetition_score", "lrs_period_repeat_count"])
    lab = pd.concat(frames, ignore_index=True)
    import numpy as np

    def _safe_nanmax(arr) -> float:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0 or bool(np.all(np.isnan(arr))):
            return float("nan")
        return float(np.nanmax(arr))

    lab["max_repetition_score"] = lab["repetition_score"].apply(_safe_nanmax)
    return lab[["prompt_id", "rollout_idx", "max_repetition_score", "lrs_period_repeat_count"]]


def build_records(
    config: DatasetGenConfig,
    backend: str,
    dataset_tag: str,
    include_heuristics: bool = True,
) -> pd.DataFrame:
    sample_path = paths.llm_judge_sample_path(config)
    results_path = paths.llm_judge_results_path(config, backend)
    if not sample_path.exists():
        raise FileNotFoundError(f"No calibration sample at {sample_path} -- run llm_judge.py first.")
    if not results_path.exists():
        raise FileNotFoundError(f"No judge results for backend {backend!r} at {results_path}.")

    sample = pd.read_parquet(sample_path)
    results = pd.read_parquet(results_path)

    truncated = sample[sample["stratum"] == "truncated"]
    ok = results[results["status"] == "ok"]
    merged = truncated.merge(ok, on=["prompt_id", "rollout_idx"], how="inner")
    if merged.empty:
        raise ValueError(
            f"No truncated rows with an 'ok' {backend!r} verdict -- nothing to export."
        )
    # Parquet round-tripping (plus the merge above) can leave this as an
    # object-dtype column of Python bools rather than numpy bool -- `~` on
    # that applies Python's integer bitwise-not (~True == -2) instead of
    # logical negation, and pandas then misreads the result as a list of
    # column labels rather than a boolean mask.
    merged["is_degenerating"] = merged["is_degenerating"].astype(bool)

    merged["onset_char_start"] = None
    merged["onset_char_end"] = None
    for idx, row in merged.iterrows():
        if isinstance(row["onset_quote"], str) and row["onset_quote"]:
            pos = row["completion_text"].find(row["onset_quote"])
            if pos != -1:
                merged.at[idx, "onset_char_start"] = int(pos)
                merged.at[idx, "onset_char_end"] = int(pos) + len(row["onset_quote"])

    if include_heuristics:
        domains = sorted(merged["domain"].unique())
        heur = _load_labels_for_heuristics(config, domains)
        merged = merged.merge(heur, on=["prompt_id", "rollout_idx"], how="left")
    else:
        merged["max_repetition_score"] = None
        merged["lrs_period_repeat_count"] = None

    merged["dataset_tag"] = dataset_tag
    merged["backend"] = backend
    return merged


def select_sample(
    records: pd.DataFrame, max_samples: Optional[int], seed: int
) -> pd.DataFrame:
    """Subsample down to ``max_samples`` rows (None = keep everything).

    Every ``is_degenerating == False`` row is kept unconditionally -- it's
    the rare, high-value population for catching false negatives (the judge
    missing real degeneration). Remaining budget is filled from the
    ``is_degenerating == True`` rows, low/medium-confidence ones first (most
    likely to be wrong), each tier shuffled with ``seed``.
    """
    if max_samples is None or len(records) <= max_samples:
        return records.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = random.Random(seed)
    negatives = records[~records["is_degenerating"]]
    positives = records[records["is_degenerating"]].copy()
    positives["_prio"] = positives["confidence"].map(_CONFIDENCE_PRIORITY).fillna(1)

    kept = [negatives]
    budget = max_samples - len(negatives)
    if budget > 0:
        tiers = []
        for _prio, group in positives.groupby("_prio"):
            idx = list(group.index)
            rng.shuffle(idx)
            tiers.extend(idx)
        kept.append(positives.loc[tiers[:budget]].drop(columns="_prio"))
    result = pd.concat(kept, ignore_index=True) if budget > 0 else negatives.reset_index(drop=True)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def assign_reviewers(records: pd.DataFrame, reviewers: List[str], seed: int) -> pd.DataFrame:
    shuffled = records.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled["assigned_to"] = [reviewers[i % len(reviewers)] for i in range(len(shuffled))]
    return shuffled


def to_review_json(records: pd.DataFrame, meta_extra: Dict[str, object]) -> Dict[str, object]:
    row_records = []
    for row in records.itertuples(index=False):
        row_records.append(
            {
                "id": f"{row.prompt_id}::{int(row.rollout_idx)}",
                "prompt_id": row.prompt_id,
                "rollout_idx": int(row.rollout_idx),
                "domain": row.domain,
                "stratum": row.stratum,
                "assigned_to": row.assigned_to,
                "prompt_text": row.prompt_text,
                "completion_text": row.completion_text,
                "heuristics": {
                    "max_repetition_score": (
                        None if pd.isna(row.max_repetition_score) else float(row.max_repetition_score)
                    ),
                    "lrs_period_repeat_count": (
                        None if pd.isna(row.lrs_period_repeat_count) else int(row.lrs_period_repeat_count)
                    ),
                },
                "judge": {
                    "backend": row.backend,
                    "is_degenerating": bool(row.is_degenerating),
                    "confidence": None if pd.isna(row.confidence) else row.confidence,
                    "reasoning": None if pd.isna(row.reasoning) else row.reasoning,
                    # Parquet round-tripping turns a column's None entries into float
                    # NaN once the column also holds real strings elsewhere (observed
                    # for onset_quote on the few is_degenerating=False rows, where the
                    # judge itself sets onset_quote=None) -- json.dump would otherwise
                    # emit the bare (non-standard) token `NaN`, which every browser's
                    # JSON.parse rejects outright.
                    "onset_quote": None if pd.isna(row.onset_quote) else row.onset_quote,
                    "onset_char_start": (
                        None if pd.isna(row.onset_char_start) else int(row.onset_char_start)
                    ),
                    "onset_char_end": (
                        None if pd.isna(row.onset_char_end) else int(row.onset_char_end)
                    ),
                },
            }
        )

    reviewer_counts: Dict[str, int] = {}
    for r in row_records:
        reviewer_counts[r["assigned_to"]] = reviewer_counts.get(r["assigned_to"], 0) + 1

    meta = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_counts": reviewer_counts,
        **meta_extra,
    }
    return {"meta": meta, "records": row_records}


def write_json_atomic(payload: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(out_path.parent), prefix=".tmp_review_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            # allow_nan=False: fail loudly here if a stray float NaN slipped through
            # instead of silently writing the bare `NaN` token, which is not valid
            # JSON and every browser's JSON.parse rejects.
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--backend", type=str, default="claude_agent_sdk")
    parser.add_argument(
        "--reviewers", type=str, required=True,
        help="Comma-separated reviewer names, e.g. 'alice,bob,carlo'.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the total number of rows exported (round-robin assigned across "
        "--reviewers after subsampling). Default: export every eligible row.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-heuristics", action="store_true",
        help="Skip loading repetition_score/LRS heuristic scores (faster, less context shown).",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Directory to write judge_review_data.json + a copy of judge_review.html into.",
    )
    args = parser.parse_args()

    reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]
    if not reviewers:
        raise ValueError("--reviewers must list at least one name.")

    config_path = Path(args.config)
    config = DatasetGenConfig.from_yaml(config_path)
    dataset_tag = config_path.stem

    records = build_records(
        config, args.backend, dataset_tag, include_heuristics=not args.skip_heuristics
    )
    print(f"Eligible rows (truncated, status=ok, backend={args.backend!r}): {len(records)}")
    print(f"  is_degenerating=True : {int(records['is_degenerating'].sum())}")
    print(f"  is_degenerating=False: {int((~records['is_degenerating']).sum())}")

    sampled = select_sample(records, args.max_samples, args.seed)
    assigned = assign_reviewers(sampled, reviewers, args.seed)
    print(f"Exporting {len(assigned)} rows across {len(reviewers)} reviewer(s):")
    print(assigned["assigned_to"].value_counts().to_string())

    payload = to_review_json(
        assigned,
        meta_extra={
            "dataset_tag": dataset_tag,
            "config_path": str(config_path),
            "backend": args.backend,
            "reviewers": reviewers,
            "max_new_tokens": config.max_new_tokens,
            "total_eligible_rows": int(len(records)),
            "sampled_rows": int(len(sampled)),
        },
    )

    out_dir = Path(args.out_dir)
    data_path = out_dir / "judge_review_data.json"
    write_json_atomic(payload, data_path)
    print(f"Wrote {data_path}")

    html_dest = out_dir / "judge_review.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_dest.write_bytes(HTML_APP_PATH.read_bytes())
    print(f"Copied {HTML_APP_PATH} -> {html_dest}")
    print()
    print("Hand out both files together (same folder) to reviewers, e.g. zip the out-dir.")


if __name__ == "__main__":
    main()
