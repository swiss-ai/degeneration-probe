"""Score a probe against human-reviewed labels instead of the judge's.

The human reviews cover the capped rollouts of the test splits. Each review
either confirms the judge, moves the onset, or overturns the verdict. This
script turns the returned review exports (plus a table of adjudications for
rows the reviewers could not settle) into labels, applies them to a run's
stored per-token scores, and reads the result at the thresholds already frozen
on validation. Nothing is re-tuned.

Four populations are reported per run and split, so a partial review round
can be read while it is still coming in:

- ``judge``: the stored labels, as in the regular evaluation.
- ``hybrid``: human labels where a review exists, judge labels elsewhere.
- ``reviewed_judge`` / ``reviewed_human``: only the rollouts that ended without
  hitting the cap plus the reviewed capped ones, with judge and human labels
  respectively. The difference between these two is the effect of the labels
  alone.

With ``--write-evaluation NAME`` the ``hybrid`` population of every run and
split is also written to ``<run-dir>/NAME/<split>/`` in the layout of the
regular ``evaluation/`` folder, so every reader of that folder (the comparison
table among them) reads the human-verified split unchanged. Once every capped
test response has a review, ``hybrid`` is the human-verified split itself.

Onset positions follow the label table's convention. A confirmed onset keeps
the stored token position. A moved onset is mapped from its character offset
to the token holding that character, then shifted by the offset between the
judge's own character onset and its stored token position on the same rollout,
which absorbs tokenizer-specific boundary effects.

    python scripts/evaluate_human_labels.py \
        --dataset-config configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml \
        --reviews human_review/round2_test_splits/returned human_review/round1_apertus_heldout \
        --adjudications human_review/round2_test_splits/returned/adjudications.csv \
        --run-dirs outputs/<run>/latest/selected/layer_15 ... \
        --out human_label_eval.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.evaluation.protocol import Threshold, evaluate
from degeneration_probe.utils.file_utils import save_json

TEST_SPLITS = ("test_indomain", "test_heldout_domains")
UNJUDGED = ("judge_failed", "not_found")
POPULATIONS = {
    "judge": (False, False),
    "hybrid": (True, False),
    "reviewed_judge": (False, True),
    "reviewed_human": (True, True),
}


def write_evaluation(out: Path, split: str, result: dict, coverage: dict) -> None:
    """One split's results in the layout of ``evaluate_scores.py``'s ``evaluation/`` folder."""
    out.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / f"{name}.csv", index=False)
    save_json(
        {
            "split": split,
            "persistence": result["persistence"],
            "thresholds": result["thresholds"],
            "rank_metrics": result["rank_metrics"],
            "labels": "human-verified where reviewed, judge elsewhere",
            "review_coverage": coverage,
        },
        out / "summary.json",
    )


def load_reviews(sources: List[Path], dataset_tag: str) -> pd.DataFrame:
    """Every review row for this dataset, one per rollout."""
    files = []
    for source in sources:
        files.extend(sorted(source.glob("judge_review_*.json")) if source.is_dir() else [source])
    rows = []
    for path in files:
        payload = json.loads(path.read_text())
        meta = payload["meta"]
        tag = meta.get("dataset_tag")
        # Round-1 held-out exports carry their own tag but were built from this
        # dataset's generations; they are passed explicitly, never globbed by tag.
        if tag != dataset_tag and tag != "heldout_degenerations":
            continue
        if tag == "heldout_degenerations" and "apertus-8b-instruct" not in dataset_tag:
            continue
        for review in payload["reviews"]:
            rows.append({**review, "reviewer": meta["reviewer"], "source_file": path.name})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    duplicated = frame[frame.duplicated("id", keep=False)]
    if len(duplicated):
        raise ValueError(f"Rollouts reviewed twice: {sorted(duplicated['id'].unique())[:10]}")
    return frame


def _int_or_none(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "None":
        return None
    return int(value)


class CharToToken:
    """Maps a character offset in a completion to the token that holds it."""

    def __init__(self, config: DatasetGenConfig, tokenizer_name: str, ids: List[str]):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        wanted = set(ids)
        frames = []
        for path in sorted((paths.dataset_root(config) / "generations").glob("*/shard_*.parquet")):
            shard = pd.read_parquet(path, columns=["prompt_id", "rollout_idx", "generated_token_ids"])
            shard["id"] = shard["prompt_id"] + "::" + shard["rollout_idx"].astype(str)
            frames.append(shard[shard["id"].isin(wanted)])
        generations = pd.concat(frames, ignore_index=True)
        self.token_ids: Dict[str, List[int]] = {
            key: [int(t) for t in tokens] for key, tokens in zip(generations["id"], generations["generated_token_ids"])
        }

    def __call__(self, rollout: str, char: int) -> int:
        ids = self.token_ids[rollout]
        low, high = 1, len(ids)
        while low < high:  # smallest k whose decoded prefix extends past `char`
            middle = (low + high) // 2
            if len(self.tokenizer.decode(ids[:middle])) > char:
                high = middle
            else:
                low = middle + 1
        return low - 1


def human_labels(
    reviews: pd.DataFrame,
    adjudications: pd.DataFrame,
    onset: pd.DataFrame,
    to_token: Optional[CharToToken],
) -> pd.DataFrame:
    """One row per reviewed rollout: keep or drop, positive or not, onset token."""
    stored = onset.set_index("id")
    decided = adjudications.set_index("id") if len(adjudications) else pd.DataFrame()
    rows = []
    for review in reviews.itertuples(index=False):
        key = review.id
        label = stored.loc[key]
        num_tokens = int(label["num_tokens"])
        judge_char = _int_or_none(review.judge_onset_char_start)
        stored_onset = None if pd.isna(label["onset_position"]) else int(label["onset_position"])

        def convert(char: int) -> int:
            shift = 0
            if judge_char is not None and stored_onset is not None:
                shift = stored_onset - to_token(key, judge_char)
            return int(min(max(to_token(key, char) + shift, 0), num_tokens - 1))

        row = {"id": key, "reviewer": review.reviewer, "split": label["split"], "drop": False,
               "is_positive": False, "onset_position": np.nan, "onset_moved": False,
               "judge_positive": bool(label["is_positive"]),
               "unjudged": label["onset_resolution"] in UNJUDGED}
        decision = decided.loc[key] if key in decided.index else None
        if label["onset_resolution"] in UNJUDGED:
            # The judge could not rule on it, so it is left out of every population, as in the judge-labelled
            # evaluation, whatever the reviewer decided.
            row["drop"] = True
        elif decision is not None and decision["decision"] == "drop":
            row["drop"] = True
        elif decision is not None and decision["decision"] == "pending":
            # Undecided: the rollout keeps whatever the stored labels say.
            continue
        elif decision is not None and decision["decision"] == "degenerating":
            row.update(is_positive=True, onset_position=convert(int(decision["onset_char"])), onset_moved=True)
        elif str(review.corrected_is_degenerating) in ("False", "false"):
            pass
        else:
            human_char = _int_or_none(review.corrected_onset_char_start)
            moved = (
                review.position_status == "corrected"
                and human_char is not None
                and human_char != judge_char
            )
            if moved:
                row.update(is_positive=True, onset_position=convert(human_char), onset_moved=True)
            elif stored_onset is not None:
                row.update(is_positive=True, onset_position=stored_onset)
            else:
                raise ValueError(f"{key}: degenerating with no onset from the reviewer or the labels")
        rows.append(row)
    return pd.DataFrame(rows)


HUMAN_LABELS = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/human_review/labels")


def labels_for_scores(scores_dir: Path) -> Optional[pd.DataFrame]:
    """The human label table of the build a scored directory describes, or None if there is none.

    The build is read from the run configuration beside the scores, the same way
    the evaluation finds its onset labels; a baseline directory names its build
    in its path."""
    from scripts.evaluate_scores import _discover_onset_labels

    onset = _discover_onset_labels(Path(scores_dir))
    names = [onset.parent.parent.name] if onset is not None else list(Path(scores_dir).parts)
    for name in names:
        dataset = name.replace("degeneration-dataset-", "")
        path = HUMAN_LABELS / f"labels_{dataset}.csv"
        if path.is_file():
            return pd.read_csv(path)
    return None


def apply_human_labels(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """A score frame with the human-verified labels: reviewed rollouts take the reviewer's verdict and onset, and
    the rollouts the reviewers dropped are removed. Rows of other splits pass through unchanged."""
    return relabel(frame, labels, human=True, reviewed_only=False)


def relabel(frame: pd.DataFrame, labels: pd.DataFrame, *, human: bool, reviewed_only: bool) -> pd.DataFrame:
    frame = frame.copy()
    frame["id"] = frame["prompt_id"] + "::" + frame["rollout_idx"].astype(str)
    reviewed = labels.set_index("id")
    if "unjudged" in reviewed.columns:
        # Responses the judge could not rule on are out of every population, judge or human, as in the
        # regular evaluation; a scorer whose score table still holds them (the baselines) drops them here.
        frame = frame[~frame["id"].isin(reviewed.index[reviewed["unjudged"].astype(bool)])]
    if reviewed_only:
        keep = (frame["stop_reason"] == "eos") | frame["id"].isin(reviewed.index)
        frame = frame[keep]
    if human:
        dropped = set(reviewed.index[reviewed["drop"]])
        frame = frame[~frame["id"].isin(dropped)].copy()
        hit = frame["id"].isin(reviewed.index)
        frame.loc[hit, "is_positive"] = reviewed.loc[frame.loc[hit, "id"], "is_positive"].to_numpy()
        frame.loc[hit, "onset_position"] = reviewed.loc[frame.loc[hit, "id"], "onset_position"].to_numpy()
    elif reviewed_only:
        # Same population as the human view, so the two differ in labels only.
        dropped = set(reviewed.index[reviewed["drop"]])
        frame = frame[~frame["id"].isin(dropped)]
    return frame.drop(columns="id").reset_index(drop=True)


def summarize(result: dict, population: str, run: str, split: str) -> List[dict]:
    coverage = result["view_b_coverage"].set_index("target_negative_fpr")
    detection = result["view_a_detection"].set_index("target_negative_fpr")
    rows = []
    for budget in coverage.index:
        rows.append({
            "run": run, "split": split, "population": population, "target_negative_fpr": budget,
            "positives": int(coverage.loc[budget, "positive_rollouts"]),
            "negatives": int(coverage.loc[budget, "negative_rollouts"]),
            "warning_recall_256": coverage.loc[budget, "warning_recall_256"],
            "in_pattern_recall": coverage.loc[budget, "in_pattern_recall"],
            "rollout_recall": detection.loc[budget, "recall"],
            "rollout_auc": result["rank_metrics"]["rollout_auc"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--tokenizer", default=None, help="Defaults to the build's model_name.")
    parser.add_argument("--reviews", type=Path, nargs="*", default=[])
    parser.add_argument("--adjudications", type=Path, default=None)
    parser.add_argument("--run-dirs", type=Path, nargs="*", default=[],
                        help="Directories holding scores/ and decision_thresholds.json (one depth each).")
    parser.add_argument("--splits", nargs="*", default=list(TEST_SPLITS))
    parser.add_argument("--labels-out", type=Path, default=None)
    parser.add_argument("--labels-in", type=Path, default=None,
                        help="A label table written earlier with --labels-out; skips reading the reviews.")
    parser.add_argument("--run-dirs-file", type=Path, default=None, help="More run directories, one per line.")
    parser.add_argument("--populations", nargs="*", default=list(POPULATIONS), choices=list(POPULATIONS))
    parser.add_argument("--write-evaluation", default=None, metavar="NAME",
                        help="Also write the hybrid population to <run-dir>/NAME/<split>/.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = DatasetGenConfig.from_yaml(args.dataset_config)
    dataset_tag = args.dataset_config.stem
    onset = pd.read_parquet(paths.onset_labels_path(config))
    onset["id"] = onset["prompt_id"] + "::" + onset["rollout_idx"].astype(str)

    if args.run_dirs_file:
        args.run_dirs = list(args.run_dirs) + [Path(line) for line in args.run_dirs_file.read_text().split() if line]
    capped = onset[onset["split"].isin(TEST_SPLITS) & (onset["stop_reason"] != "eos")]
    if args.labels_in:
        labels = pd.read_csv(args.labels_in)
        coverage = {split: {"reviewed": int((labels["split"] == split).sum()),
                            "capped": int((capped["split"] == split).sum())} for split in TEST_SPLITS}
        print(f"{dataset_tag}: labels from {args.labels_in}, {coverage}")
    else:
        labels, coverage = build_labels(args, config, dataset_tag, onset, capped)
    evaluate_runs(args, labels, coverage)


def build_labels(args, config, dataset_tag: str, onset: pd.DataFrame, capped: pd.DataFrame):
    reviews = load_reviews(args.reviews, dataset_tag)
    adjudications = pd.read_csv(args.adjudications) if args.adjudications else pd.DataFrame()
    if len(adjudications):
        adjudications = adjudications[adjudications["model"] == dataset_tag.replace("degeneration-dataset-", "")]
    print(f"{dataset_tag}: {len(reviews)} reviews over {len(capped)} capped test rollouts")
    coverage = {}
    for split in TEST_SPLITS:
        in_split = set(capped.loc[capped["split"] == split, "id"])
        coverage[split] = {"reviewed": len(in_split & set(reviews["id"])), "capped": len(in_split)}
        print(f"  {split}: {coverage[split]['reviewed']}/{coverage[split]['capped']} reviewed")
    stray = set(reviews["id"]) - set(capped["id"])
    if stray:
        raise ValueError(f"Reviews for rollouts outside the capped test population: {sorted(stray)[:5]}")

    moved = reviews[
        (reviews["position_status"] == "corrected")
        | reviews["id"].isin(adjudications.get("id", pd.Series(dtype=str)))
    ]["id"].tolist()
    to_token = CharToToken(config, args.tokenizer or config.model_name, moved) if moved else None
    labels = human_labels(reviews, adjudications, onset, to_token)
    flips = labels[~labels["drop"] & (labels["is_positive"] != labels["judge_positive"])]
    print(f"  labels: {int(labels['drop'].sum())} dropped, {int(labels['onset_moved'].sum())} onsets moved, "
          f"{len(flips)} verdicts differ from the stored labels")
    if args.labels_out:
        args.labels_out.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(args.labels_out, index=False)
    return labels, coverage


def evaluate_runs(args, labels: pd.DataFrame, coverage: dict) -> None:
    populations = list(args.populations)
    if args.write_evaluation and "hybrid" not in populations:
        populations.append("hybrid")

    rows = []
    for run_dir in args.run_dirs:
        thresholds_payload = json.loads((run_dir / "decision_thresholds.json").read_text())
        thresholds = [Threshold(**t) for t in thresholds_payload["thresholds"]]
        persistence = int(thresholds_payload["persistence"])
        name = "/".join(run_dir.parts[-5:])
        for split in args.splits:
            frame = pd.read_parquet(run_dir / "scores" / f"{split}.parquet")
            frame["scores"] = [np.asarray(v, dtype=np.float32) for v in frame["scores"]]
            unscored = set(labels.loc[(labels["split"] == split) & ~labels["drop"], "id"]) - set(
                frame["prompt_id"] + "::" + frame["rollout_idx"].astype(str)
            )
            if unscored:
                print(f"  {split}: {len(unscored)} reviewed rollouts have no scores (judge failed at scoring time)")
            for population in populations:
                human, reviewed_only = POPULATIONS[population]
                view = relabel(frame, labels, human=human, reviewed_only=reviewed_only)
                result = evaluate(view, thresholds, persistence=persistence)
                rows.extend(summarize(result, population, name, split))
                if population == "hybrid" and args.write_evaluation:
                    write_evaluation(run_dir / args.write_evaluation / split, split, result, coverage[split])
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    at_one = out[out["target_negative_fpr"] == 0.01]
    print(at_one.groupby(["split", "population"])[["positives", "negatives", "warning_recall_256", "rollout_recall"]]
          .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
