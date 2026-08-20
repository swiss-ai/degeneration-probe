"""Re-measure a per-token scorer at the units other papers report on.

The nearest prior work does not all measure at the same granularity, and the
difference matters for what its headline numbers can be read as saying:

    sentence   a sentence's hidden states are averaged and the sentence is
               classified
    chunk      a reasoning trace is cut at every blank line and the hidden state
               of the token ending each chunk is classified
    answer     a whole response is classified

Given per-token scores from the protocol, this module cuts each answer into units
of each kind, reduces the tokens of a unit to one number, and reports how well
the result separates units inside a loop from units that are not. A unit is
positive when it begins at or after the frontier, so the notion of "inside the
loop" is the same one the rest of the project uses.

Two limits are worth stating, because the units are matched to those papers and
the *operators* are not.

The chunk reading is faithful: a single token's representation decides a chunk,
which is what a probe already does. The sentence reading is a proxy. Averaging a
probe's scores over a sentence is not the same as classifying the average of that
sentence's hidden states, because the probe applies a normalisation and a
sigmoid, and neither commutes with the mean. The answer reading is this project's
own rule, a maximum over tokens, not the trained network over sorted similarity
features that a response-level detector in the literature uses.

So this measures how the reported quantity behaves at each unit *size*. It does
not reimplement anybody's detector, and a row here is not a reproduction of a
published number.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from degeneration_probe.evaluation.scores import drop_unjudged

# A sentence ends at a run of terminal punctuation followed by whitespace or by
# the end of the text. Deliberately crude. A real sentence splitter would
# disagree with this one on abbreviations and decimals, and those disagreements
# would need their own investigation; the question here is the size of a unit,
# not where a linguist would cut it.
SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
# A chunk boundary is a blank line, which is what a double newline is once any
# trailing spaces on the first line are allowed for.
CHUNK_BREAK = re.compile(r"\n\s*\n")
# The delimiters above are all ASCII and are never split across tokens. That is
# what makes the boundaries reliable even though a piecewise decode mangles
# multi-byte characters, and it is checked rather than assumed.
DELIMITERS = ("\n", ".", "!", "?")
BALANCED_SEED = 42

# name, how a unit's score is reduced from its tokens, whose granularity it is
UNITS: Tuple[Tuple[str, str, str], ...] = (
    ("sentence", "mean", "Duan et al."),
    ("chunk", "final", "Xie et al."),
    ("answer", "max", "Yu et al."),
)


def token_char_spans(tokenizer, token_ids: Sequence[int]) -> Tuple[str, np.ndarray]:
    """The decoded text, and the character each token starts at.

    Built by decoding tokens one at a time and concatenating, so the text and the
    offsets are consistent with each other by construction. A boundary found in
    this string therefore maps back to a token exactly.

    It is not byte-identical to the stored generation: a character outside ASCII
    is several bytes and can be split across tokens, and each piece then decodes
    to a replacement character on its own. Only the delimiters matter here and
    they are all ASCII, which `delimiters_survive` checks.
    """
    pieces = [tokenizer.decode([int(token)]) for token in token_ids]
    starts = np.zeros(len(pieces), dtype=np.int64)
    position = 0
    for index, piece in enumerate(pieces):
        starts[index] = position
        position += len(piece)
    return "".join(pieces), starts


def delimiters_survive(piecewise: str, stored: str) -> bool:
    """Whether every unit delimiter came through the piecewise decode intact."""
    return all(piecewise.count(mark) == stored.count(mark) for mark in DELIMITERS)


def _token_of_char(starts: np.ndarray, char: int) -> int:
    """The token whose piece contains a character position."""
    index = int(np.searchsorted(starts, char, side="right") - 1)
    return max(0, min(index, starts.size - 1))


def unit_boundaries(text: str, starts: np.ndarray, kind: str) -> List[Tuple[int, int]]:
    """Inclusive token spans for one segmentation of one answer.

    Spans tile the answer with no gaps: each ends at the token holding the last
    character of its delimiter, the next begins at the token after, and whatever
    follows the final delimiter forms a last span. So every token belongs to
    exactly one unit and none is silently dropped.
    """
    total = starts.size
    if total == 0:
        return []
    if kind == "answer":
        return [(0, total - 1)]
    if kind == "chunk":
        pattern = CHUNK_BREAK
    elif kind == "sentence":
        pattern = SENTENCE_END
    else:
        raise ValueError(f"unknown unit {kind!r}")

    ends = [_token_of_char(starts, match.end() - 1) for match in pattern.finditer(text)]
    spans: List[Tuple[int, int]] = []
    start = 0
    for end in ends:
        if end < start:
            continue
        spans.append((start, end))
        start = end + 1
    if start <= total - 1:
        spans.append((start, total - 1))
    return spans


def unit_scores(
    scores: np.ndarray, spans: Sequence[Tuple[int, int]], reduce: str
) -> np.ndarray:
    """One score per unit, reduced the way the matched granularity reduces it."""
    if reduce == "final":
        return np.array([scores[end] for _, end in spans], dtype=np.float64)
    if reduce == "mean":
        return np.array(
            [scores[start : end + 1].mean() for start, end in spans], dtype=np.float64
        )
    if reduce == "max":
        return np.array(
            [scores[start : end + 1].max() for start, end in spans], dtype=np.float64
        )
    raise ValueError(f"unknown reduction {reduce!r}")


def roc_auc(labels: np.ndarray, values: np.ndarray) -> float:
    """Rank AUC, with tied scores sharing the average rank."""
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float(
        (ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
    )


def _descending_groups(labels: np.ndarray, values: np.ndarray):
    """Cumulative positive and negative counts at each distinct score, descending.

    Returns arrays of (true positives, false positives) for the rule "fire when the
    score is at or above this value", one entry per distinct value. Ties are kept
    together because no threshold can separate them, and treating them as separable
    silently flatters every metric built on a cut point.
    """
    order = np.argsort(-values, kind="mergesort")
    ranked, sorted_values = labels[order], values[order]
    boundaries = np.flatnonzero(np.r_[np.diff(sorted_values) != 0, True])
    true_positive = np.cumsum(ranked == 1)[boundaries]
    false_positive = np.cumsum(ranked == 0)[boundaries]
    return true_positive, false_positive


def best_accuracy(labels: np.ndarray, values: np.ndarray) -> float:
    """The highest accuracy any threshold on this score achieves, as measured.

    On the natural population rather than a balanced subsample, so it carries the
    class balance with it. That is the point of reporting it: at answer level the
    positive rate is about 3%, so refusing to fire at all already scores 0.97, and
    the distance between that floor and a probe is the whole dynamic range the
    quantity has.
    """
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    true_positive, false_positive = _descending_groups(labels, values)
    # Firing on nothing is a candidate too, and at answer level it is a strong one.
    correct = np.append(true_positive + (negative - false_positive), negative)
    return float(correct.max() / labels.size)


def average_precision(labels: np.ndarray, values: np.ndarray) -> float:
    """Area under the precision-recall curve, which class imbalance does move.

    Reported beside the area under the ROC curve because the two disagree exactly
    where a population is skewed, and this one is: at answer level the positives are
    3% of the split.
    """
    positive = int((labels == 1).sum())
    if positive == 0 or positive == labels.size:
        return float("nan")
    true_positive, false_positive = _descending_groups(labels, values)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / positive
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def balanced_accuracy(
    labels: np.ndarray, values: np.ndarray, seed: int = BALANCED_SEED
) -> float:
    """Best accuracy achievable on an equal-sized subsample of the two classes.

    The threshold is chosen on the same subsample it is scored on, which is
    generous. That is deliberate: the figure this sits beside is one a trained
    classifier reports on its own balanced test set, and the argument is that even
    the most flattering reading of it barely moves between scorers.
    """
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    size = min(positive.size, negative.size)
    generator = np.random.default_rng(seed)
    chosen = np.concatenate(
        [
            generator.choice(positive, size, replace=False),
            generator.choice(negative, size, replace=False),
        ]
    )
    truth, score = labels[chosen], values[chosen]
    true_positive, false_positive = _descending_groups(truth, score)
    negatives = int((truth == 0).sum())
    correct = np.append(true_positive + (negatives - false_positive), negatives)
    return float(correct.max() / truth.size)


def segment_split(
    build_root: Union[str, Path], tokenizer_path: Union[str, Path], split: str
) -> Dict[str, dict]:
    """Token spans for every answer in a split, computed once and reused.

    Keyed by prompt and rollout index, so a scorer's table is joined to it by
    identity rather than by row order.
    """
    from tokenizers import Tokenizer

    build_root = Path(build_root)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    onset = pd.read_parquet(build_root / "onset_labels" / "onset_labels.parquet")
    onset = onset[onset["split"] == split]

    segmented: Dict[str, dict] = {}
    unreliable = 0
    for domain in sorted(onset["domain"].astype(str).unique()):
        generations = pd.concat(
            [
                pd.read_parquet(
                    path,
                    columns=[
                        "prompt_id",
                        "rollout_idx",
                        "generated_token_ids",
                        "generated_text",
                    ],
                )
                for path in sorted((build_root / "generations" / domain).glob("*.parquet"))
            ],
            ignore_index=True,
        )
        merged = onset[onset["domain"].astype(str) == domain].merge(
            generations, on=["prompt_id", "rollout_idx"], how="inner", validate="one_to_one"
        )
        for row in merged.itertuples(index=False):
            ids = list(row.generated_token_ids)[: int(row.num_tokens)]
            text, starts = token_char_spans(tokenizer, ids)
            if not delimiters_survive(text, row.generated_text):
                unreliable += 1
            segmented[f"{row.prompt_id}:{row.rollout_idx}"] = {
                "num_tokens": int(row.num_tokens),
                "onset": int(row.onset_position) if pd.notna(row.onset_position) else None,
                "is_positive": bool(row.is_positive) and pd.notna(row.onset_position),
                "spans": {kind: unit_boundaries(text, starts, kind) for kind, _, _ in UNITS},
            }
    segmented["__unreliable__"] = {"count": unreliable}
    return segmented


def measure(
    scores_path: Union[str, Path],
    segmented: Dict[str, dict],
    unjudged: Optional[set] = None,
) -> List[dict]:
    """One row per unit definition for one scorer's stored per-token scores.

    ``unjudged`` drops the rollouts nothing is known about, so that a table holding
    both probes and model-free scorers compares them on one population. Probe score
    files already leave them out, because the pinned evaluation population does;
    the baselines' files do not, and without this a single table would mix 3,634
    answers with 3,640.
    """
    frame = pd.read_parquet(scores_path)
    if unjudged:
        frame, _ = drop_unjudged(frame, unjudged)
    collected: Dict[str, Tuple[List, List]] = {kind: ([], []) for kind, _, _ in UNITS}
    for row in frame.itertuples(index=False):
        record = segmented.get(f"{row.prompt_id}:{row.rollout_idx}")
        if record is None:
            continue
        values = np.asarray(row.scores, dtype=np.float64)
        for kind, reduce, _ in UNITS:
            spans = record["spans"][kind]
            if not spans:
                continue
            if kind == "answer":
                labels = np.array([1 if record["is_positive"] else 0])
            elif record["is_positive"]:
                labels = np.array(
                    [1 if start >= record["onset"] else 0 for start, _ in spans]
                )
            else:
                labels = np.zeros(len(spans), dtype=int)
            collected[kind][0].append(unit_scores(values, spans, reduce))
            collected[kind][1].append(labels)

    rows = []
    for kind, _, whose in UNITS:
        values = (
            np.concatenate(collected[kind][0]) if collected[kind][0] else np.zeros(0)
        )
        labels = (
            np.concatenate(collected[kind][1])
            if collected[kind][1]
            else np.zeros(0, dtype=int)
        )
        rows.append(
            {
                "unit": kind,
                "granularity_of": whose,
                "answers": int(frame.shape[0]),
                "units": int(values.size),
                "positive_units": int((labels == 1).sum()),
                "positive_rate": float((labels == 1).mean()) if values.size else np.nan,
                "auc": roc_auc(labels, values),
                "average_precision": average_precision(labels, values),
                "accuracy": best_accuracy(labels, values),
                "accuracy_floor": float(max((labels == 0).mean(), (labels == 1).mean())),
                "balanced_accuracy": balanced_accuracy(labels, values),
            }
        )
    return rows
