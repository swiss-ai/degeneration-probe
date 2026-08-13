"""Average probe score as a function of distance from the degeneration frontier.

Coverage and lead time together imply something about the shape of a probe's
score over a rollout: a rule that fires early but covers little should be
reading a spike at onset, and a rule that fires late but covers a lot should be
reading a plateau that lasts. That is an inference, and this figure measures the
thing directly instead.

Every positive rollout is aligned on its own frontier and the scores are
averaged across rollouts and seeds at each offset, so position zero is the token
where degeneration begins for every rollout in the average.

    python scripts/plot_score_trajectories.py --layer 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The reference categorical palette, in fixed slot order. Hues are assigned by
# position in this list and never cycled, so a rule keeps its colour whatever
# else is on the axes.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"

RUNGS = [
    ("all_tokens", "all tokens"),
    ("rollout_balanced", "rollout balanced"),
    ("random_window", "random window"),
    ("frontier_window", "frontier window"),
    ("frontier_window_hard_negative", "frontier + hard neg."),
]


def find_runs(layer: int):
    """Every equal-budget ladder run, grouped by its selection rule."""
    found = {}
    for config_path in sorted(Path("outputs").glob("*/*/resolved_config.json")):
        if config_path.parent.is_symlink():
            continue
        config = json.loads(config_path.read_text())
        training = config["training"]
        selection = training.get("selection")
        if not selection or training["probe"].get("layers") is None:
            continue
        if training["runtime"]["max_steps"] != 800 or selection["window_size"] != 128:
            continue
        scores = config_path.parent / "layers" / f"layer_{layer:02d}" / "scores" / "val.parquet"
        if scores.is_file():
            found.setdefault(selection["strategy"], []).append(config_path.parent)
    return found


def trajectory(run_dirs, layer: int, span: tuple[int, int], *, negatives=False, seed=0):
    """Mean score and mean share above threshold, by offset from the frontier.

    A rising score before the frontier means nothing on its own, because a score
    could drift upward with position for reasons that have nothing to do with
    degeneration. Healthy rollouts are therefore aligned the same way, on a
    pseudo-frontier drawn from the frontier distribution of the degenerate ones,
    which gives the same axis a null to read the rise against.
    """
    low, high = span
    width = high - low
    score_total = np.zeros(width)
    alarm_total = np.zeros(width)
    counts = np.zeros(width)
    rng = np.random.default_rng(seed)

    for run_dir in run_dirs:
        base = run_dir / "layers" / f"layer_{layer:02d}"
        frame = pd.read_parquet(base / "scores" / "val.parquet")
        thresholds = json.loads((base / "decision_thresholds.json").read_text())
        # The strictest budget, which is the operating point a monitor would run.
        tau = min(t["tau"] for t in thresholds["thresholds"])

        positives = frame[frame.is_positive]
        onsets = positives["onset_position"].to_numpy(dtype=float)
        rows = frame[~frame.is_positive] if negatives else positives

        for row in rows.itertuples():
            scores = np.asarray(row.scores, dtype=np.float32)
            if negatives:
                # A healthy rollout has no frontier, so it borrows one. Only
                # draws that land inside the rollout are usable.
                usable = onsets[onsets < len(scores)]
                if not usable.size:
                    continue
                onset = int(rng.choice(usable))
            else:
                onset = int(row.onset_position)
            # Where this rollout's own token range lands on the shared axis.
            start = max(low, -onset)
            stop = min(high, len(scores) - onset)
            if stop <= start:
                continue
            piece = scores[onset + start : onset + stop]
            index = slice(start - low, stop - low)
            score_total[index] += piece
            alarm_total[index] += piece >= tau
            counts[index] += 1

    counts = np.where(counts == 0, np.nan, counts)
    return score_total / counts, alarm_total / counts, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--before", type=int, default=512, help="Tokens of run-up to show.")
    parser.add_argument("--after", type=int, default=1024, help="Tokens of loop to show.")
    parser.add_argument("--min-rollouts", type=int, default=30)
    args = parser.parse_args()

    span = (-args.before, args.after)
    offsets = np.arange(*span)
    found = find_runs(args.layer)
    if not found:
        raise SystemExit(f"No scored ladder run at layer {args.layer}.")

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    drawn = []
    for slot, (strategy, label) in enumerate(RUNGS):
        run_dirs = found.get(strategy)
        if not run_dirs:
            print(f"  {strategy}: no scored run, skipped")
            continue
        score, alarm, counts = trajectory(run_dirs, args.layer, span)
        # Only where enough rollouts are still long enough to contribute.
        thin = counts < args.min_rollouts * len(run_dirs)
        score[thin] = np.nan
        alarm[thin] = np.nan
        colour = SERIES[slot]
        axes[0].plot(offsets, score, color=colour, lw=1.8, label=label)
        axes[1].plot(offsets, alarm, color=colour, lw=1.8, label=label)

        # The same rule on healthy rollouts: the null this rise is read against.
        null_score, null_alarm, null_counts = trajectory(
            run_dirs, args.layer, span, negatives=True
        )
        thin_null = null_counts < args.min_rollouts * len(run_dirs)
        null_score[thin_null] = np.nan
        null_alarm[thin_null] = np.nan
        axes[0].plot(offsets, null_score, color=colour, lw=1.0, ls=(0, (2, 2)), alpha=0.75)
        axes[1].plot(offsets, null_alarm, color=colour, lw=1.0, ls=(0, (2, 2)), alpha=0.75)

        drawn.append((label, colour, score, alarm, null_score, null_alarm))
        print(f"  {strategy}: {len(run_dirs)} run(s)")

    for axis, title, ylabel in (
        (axes[0], "What the probe scores", "mean probe score"),
        (axes[1], "How often it is above threshold", "share of rollouts alarming"),
    ):
        # The frontier itself: recessive, because it is a reference and not data.
        axis.axvline(0, color=INK_SOFT, lw=1, ls=(0, (4, 3)))
        axis.annotate(
            "frontier",
            xy=(0, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -10),
            textcoords="offset points",
            color=INK_SOFT,
            fontsize=8,
        )
        axis.set_title(title, color=INK, fontsize=11, loc="left")
        axis.set_xlabel("tokens from the frontier (negative is before)", color=INK_SOFT, fontsize=9)
        axis.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)
        axis.grid(True, color=GRID, lw=0.6, alpha=0.9)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=INK_SOFT, labelsize=8)
        axis.set_xlim(*span)

    axes[1].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    # The solid/dashed distinction carries as much meaning as the colour, so it
    # is named in the legend rather than left to be inferred.
    style_key = [
        matplotlib.lines.Line2D([], [], color=INK_SOFT, lw=1.8),
        matplotlib.lines.Line2D([], [], color=INK_SOFT, lw=1.0, ls=(0, (2, 2))),
    ]
    legend = figure.legend(
        handles + style_key,
        labels + ["degenerate", "healthy (pseudo-frontier)"],
        loc="lower center",
        ncol=len(labels) + 2,
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    figure.suptitle(
        f"Probe score around the degeneration frontier, layer {args.layer}, validation",
        color=INK,
        fontsize=12,
        x=0.008,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.94))

    out = REPO_ROOT / "notebooks" / "figures" / "diary"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"score_trajectory_L{args.layer:02d}"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"\nWrote {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")

    # The numbers behind the shape, so the figure can be read without pixels.
    # What matters is not how high a curve sits before the frontier but how far
    # it sits above the same rule's healthy curve: a rise shared with healthy
    # rollouts separates nothing, however early it starts.
    marks = [-512, -256, -128, -64, 0, 128]
    at = lambda series, offset: series[int(np.searchsorted(offsets, offset))]

    print("\nMean score, degenerate rollouts:")
    print(f"{'rule':<24}" + "".join(f"{m:>+9d}" for m in marks))
    for label, _, score, _, _, _ in drawn:
        print(f"{label:<24}" + "".join(f"{at(score, m):>9.3f}" for m in marks))

    print("\nSeparation from the same rule on healthy rollouts (degenerate - healthy):")
    print(f"{'rule':<24}" + "".join(f"{m:>+9d}" for m in marks))
    for label, _, score, _, null_score, _ in drawn:
        print(
            f"{label:<24}"
            + "".join(f"{at(score, m) - at(null_score, m):>9.3f}" for m in marks)
        )


if __name__ == "__main__":
    main()
