"""Move from this protocol to the one the closest prior work uses, one choice at a time.

Reported early-warning numbers in this area are not comparable, and the reasons
are usually listed rather than measured. Three of those reasons are decisions
about how a score becomes a decision, not about the scorer itself, so they can be
turned on scores that already exist:

- **How token evidence is aggregated.** Deciding per token, deciding on a pooled
  span, or accumulating span evidence with a cumulative-sum rule.
- **Which operating point is read.** A budget of one percent of healthy answers,
  or the twenty-five to thirty-five percent that a balanced-benchmark setting can
  afford.
- **What the base rate is.** Natural prevalence, or an evaluation set rebalanced
  to equal numbers of degenerate and healthy answers.

Holding one scorer fixed and turning these in sequence gives a ladder from one
protocol to another, and the distance between rungs is the conversion factor the
literature is missing. Nothing here evaluates anyone else's detector: it is this
scorer read under other people's reporting choices, and it says nothing about
what their model would do on this data.

    python scripts/protocol_bridge.py --scores <run>/layers/layer_15/scores/val.parquet

Span pooling stands in for sentence pooling. The scores carry no text, so a span
is a fixed number of tokens rather than a sentence, which also makes the span
length a knob that can be swept instead of a property of the corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.protocol import coverage_window, threshold_for_budget

WARNING_BAND = 256
DEFAULT_SPAN = 32
DEFAULT_BUDGETS = (0.01, 0.05, 0.10, 0.30)


def aggregate(scores: np.ndarray, mode: str, span: int, drift: float = 0.0) -> np.ndarray:
    """Turn per-token scores into the statistic a given protocol decides on.

    Every mode returns one value per token, so the metrics downstream stay token
    resolved and the three modes are read on one axis.

    A pooled decision is only available once its span has finished, so a span's
    value is carried from its final token rather than from its first. Without
    that shift a span detector is credited with firing up to a span early, which
    is exactly the quantity being measured. A rollout shorter than one span
    therefore never fires, which is the honest behaviour of a detector that has
    to see a whole span before it decides.
    """
    values = np.asarray(scores, dtype=np.float64)
    if mode == "token":
        return values
    length = values.size
    spans = int(np.ceil(length / span)) if length else 0
    if spans == 0:
        return values
    padded = np.full(spans * span, np.nan)
    padded[:length] = values
    with np.errstate(invalid="ignore"):
        pooled = np.nanmean(padded.reshape(spans, span), axis=1)

    if mode == "span_cusum":
        # The usual cumulative-sum recursion: evidence above a reference drift
        # accumulates, evidence below it decays, and the total is floored at zero.
        total = 0.0
        accumulated = np.empty_like(pooled)
        for index, value in enumerate(pooled):
            total = max(0.0, total + (value - drift))
            accumulated[index] = total
        pooled = accumulated
    elif mode != "span":
        raise ValueError(f"unknown aggregation {mode!r}")

    broadcast = np.repeat(pooled, span)
    shifted = np.concatenate([np.full(span - 1, -np.inf), broadcast])[:length]
    return shifted


def reference_drift(frame: pd.DataFrame, span: int) -> float:
    """The level a cumulative sum treats as no evidence, from healthy answers."""
    healthy = frame[~frame["is_positive"].astype(bool)]
    pooled = [aggregate(s, "span", span) for s in healthy["scores"]]
    finite = np.concatenate([p[np.isfinite(p)] for p in pooled if np.isfinite(p).any()])
    return float(np.median(finite))


def rung_metrics(
    frame: pd.DataFrame,
    *,
    mode: str,
    span: int,
    budget: float,
    drift: float,
    negative_index: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Every number one protocol choice produces, at one operating point."""
    positives = frame[frame["is_positive"].astype(bool)]
    negatives = frame[~frame["is_positive"].astype(bool)]
    if negative_index is not None:
        negatives = negatives.iloc[negative_index]

    def peak(scores):
        values = aggregate(scores, mode, span, drift)
        return float(values.max()) if values.size else -np.inf

    negative_peaks = np.array([peak(s) for s in negatives["scores"]])
    tau, realized = threshold_for_budget(negative_peaks, budget)

    caught = 0
    inside_hits = inside_total = band_hits = band_total = 0
    offsets: List[float] = []
    for scores, onset in zip(positives["scores"], positives["onset_position"]):
        values = aggregate(scores, mode, span, drift)
        onset = int(onset)
        above = values >= tau
        if above.any():
            caught += 1
            offsets.append(float(np.flatnonzero(above)[0] - onset))
        inside = above[coverage_window(values.size, onset, None)]
        inside_hits += int(inside.sum())
        inside_total += inside.size
        band = above[coverage_window(values.size, onset, WARNING_BAND)]
        band_hits += int(band.sum())
        band_total += band.size

    false_alarms = int((negative_peaks >= tau).sum())
    flagged = caught + false_alarms
    offsets_array = np.asarray(offsets)
    return {
        "recall": caught / len(positives),
        "precision": (caught / flagged) if flagged else float("nan"),
        "realized_fpr": realized,
        "in_pattern": inside_hits / inside_total if inside_total else float("nan"),
        f"warning_{WARNING_BAND}": band_hits / band_total if band_total else float("nan"),
        "median_offset": float(np.median(offsets_array)) if offsets_array.size else float("nan"),
        "fired_before": float((offsets_array < 0).mean()) if offsets_array.size else float("nan"),
        "never_fired": len(positives) - caught,
    }


def balanced_draws(frame: pd.DataFrame, draws: int, seed: int) -> List[np.ndarray]:
    """Index sets that thin the healthy answers to the number of degenerate ones."""
    positives = int(frame["is_positive"].astype(bool).sum())
    negatives = int((~frame["is_positive"].astype(bool)).sum())
    rng = np.random.default_rng(seed)
    return [rng.choice(negatives, size=min(positives, negatives), replace=False) for _ in range(draws)]


def averaged(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}


def build(frame: pd.DataFrame, span: int, budgets: List[float], draws: int, seed: int) -> pd.DataFrame:
    drift = reference_drift(frame, span)
    low, high = budgets[0], budgets[-1]
    balanced = balanced_draws(frame, draws, seed)

    def run(mode, budget, base_rate):
        if base_rate == "natural":
            return rung_metrics(frame, mode=mode, span=span, budget=budget, drift=drift)
        return averaged(
            [
                rung_metrics(frame, mode=mode, span=span, budget=budget, drift=drift, negative_index=index)
                for index in balanced
            ]
        )

    ladder = [
        ("0. this protocol", "token", low, "natural"),
        ("1. + span pooling", "span", low, "natural"),
        ("2. + cumulative sum", "span_cusum", low, "natural"),
        ("3. + their operating point", "span_cusum", high, "natural"),
        ("4. + balanced base rate", "span_cusum", high, "balanced"),
    ]
    one_at_a_time = [
        ("span pooling only", "span", low, "natural"),
        ("cumulative sum only", "span_cusum", low, "natural"),
        ("operating point only", "token", high, "natural"),
        ("balanced base rate only", "token", low, "balanced"),
    ]

    rows = []
    for table, entries in (("ladder", ladder), ("one at a time", one_at_a_time)):
        for label, mode, budget, base_rate in entries:
            rows.append(
                {
                    "table": table,
                    "step": label,
                    "aggregation": mode,
                    "budget": budget,
                    "base rate": base_rate,
                    **run(mode, budget, base_rate),
                }
            )
    frame_out = pd.DataFrame(rows)
    frame_out.attrs["drift"] = drift
    return frame_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="A per-token scores parquet.")
    parser.add_argument("--span", type=int, default=DEFAULT_SPAN, help="Tokens per pooled span.")
    parser.add_argument("--budgets", nargs="*", type=float, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--draws", type=int, default=20, help="Rebalanced draws to average over.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Write the table as CSV.")
    args = parser.parse_args()

    frame = pd.read_parquet(args.scores)
    positives = int(frame["is_positive"].astype(bool).sum())
    print(f"{args.scores}")
    print(f"  {len(frame)} answers, {positives} degenerate, span {args.span} tokens\n")

    table = build(frame, args.span, sorted(args.budgets), args.draws, args.seed)
    pd.set_option("display.width", 220)
    columns = [
        "step", "recall", "precision", "realized_fpr", "in_pattern",
        f"warning_{WARNING_BAND}", "median_offset", "fired_before", "never_fired",
    ]
    for name, block in table.groupby("table", sort=False):
        print(f"=== {name} ===")
        print(block[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()
    print(f"cumulative-sum reference drift, from healthy spans: {table.attrs['drift']:.4f}")
    if args.output:
        table.to_csv(args.output, index=False)
        print(f"written to {args.output}")


if __name__ == "__main__":
    main()
