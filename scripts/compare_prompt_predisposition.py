"""Does a probe read "this trajectory is breaking" or "this prompt is the kind
that breaks"?

Every prompt in this corpus has ten rollouts. Some prompts never produce a
degenerate rollout (clean); others produce at least one alongside otherwise
healthy siblings (loop-prone). Nothing went wrong in a loop-prone prompt's
healthy sibling -- it finished on its own, same as a clean prompt's rollout.
A probe reading imminence should score the two the same. A probe that has
instead picked up on prompt-level difficulty -- topic, phrasing, whatever
correlates with a prompt looping at all -- will score the loop-prone prompt's
healthy siblings higher, because a prompt's difficulty is fixed at sampling
time and has nothing to do with what a specific trajectory did.

This only works because the corpus keeps ten rollouts per prompt: without a
finished sibling next to a degenerate one there is no clean control for what
the same prompt "should" have looked like.

Two numbers carry the answer. The rank-biserial AUC (restricted to healthy
rollouts only, label = does this rollout's prompt have a degenerate sibling)
is the probe's ability to guess a prompt's predisposition from a trajectory
that never showed any sign of trouble -- 0.5 means it can't, 1.0 means it can
perfectly. And the alarm rate at each of the project's own frozen thresholds,
split by the same grouping, is what that leakage costs in practice: false
alarms landing disproportionately on loop-prone-prompt siblings is the
signature of a predisposition detector wearing an imminence detector's
threshold.

    python scripts/compare_prompt_predisposition.py \\
        --scores outputs/<frozen_run>/layers/layer_15/scores/val.parquet frozen \\
        --scores outputs/<lora_run>/scores/val.parquet lora-adapted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from degeneration_probe.evaluation.protocol import choose_thresholds, rollout_score  # noqa: E402
from degeneration_probe.evaluation.scores import read_scores  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"

DEFAULT_LABELS = (
    Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
    / "degeneration-dataset-apertus-8b-instruct" / "onset_labels" / "onset_labels.parquet"
)


def prompt_predisposition(labels_path: Path, split: str) -> pd.Series:
    """Which prompts in this split ever produced a degenerate rollout.

    Read from the raw label file rather than the scores table, so the flag is
    the same whichever run's scores are being read: it is a property of the
    prompt and the sampling that generated its rollouts, not of any probe.
    """
    labels = pd.read_parquet(
        labels_path, columns=["prompt_id", "split", "is_positive"]
    )
    labels = labels[labels["split"] == split]
    if labels.empty:
        raise ValueError(f"No rows for split {split!r} in {labels_path}")
    return labels.groupby("prompt_id")["is_positive"].any()


def healthy_groups(
    frame: pd.DataFrame, loop_prone: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Healthy rollouts split by whether their own prompt ever looped.

    Healthy means it finished on its own -- stop_reason == 'eos' -- rather
    than merely is_positive == False, which would also admit the handful of
    capped rollouts the judge could not resolve. Those are not a clean
    trajectory to compare against; they are dropped from both groups.
    """
    healthy = frame[frame["stop_reason"] == "eos"].copy()
    healthy["prompt_loop_prone"] = healthy["prompt_id"].map(loop_prone)
    missing = healthy["prompt_loop_prone"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} healthy rollouts have a prompt_id absent from "
            "the label file -- labels and scores disagree on which corpus this is."
        )
    healthy["prompt_loop_prone"] = healthy["prompt_loop_prone"].astype(bool)
    siblings = healthy[healthy["prompt_loop_prone"]]
    clean = healthy[~healthy["prompt_loop_prone"]]
    return siblings, clean


def rank_biserial_auc(siblings: np.ndarray, clean: np.ndarray) -> Tuple[float, float]:
    """P(a random sibling scores above a random clean rollout), and its p-value.

    This is the whole test as one number. 0.5 is what an imminence-only probe
    should produce on trajectories that never showed anything -- the two
    populations are drawn from the same distribution of "finished fine" token
    sequences and differ only in a fact about the prompt the probe never
    reads. Anything reliably above 0.5 is the probe using that fact anyway.
    """
    statistic, p_value = stats.mannwhitneyu(siblings, clean, alternative="greater")
    auc = float(statistic / (len(siblings) * len(clean)))
    return auc, float(p_value)


def alarm_rates(
    siblings: pd.DataFrame, clean: pd.DataFrame, thresholds
) -> pd.DataFrame:
    """The false-alarm rate at each frozen operating point, split by group."""
    rows = []
    for threshold in thresholds:
        sib_fired = (siblings["rollout_score"] >= threshold.tau).mean()
        clean_fired = (clean["rollout_score"] >= threshold.tau).mean()
        rows.append(
            {
                "target_negative_fpr": threshold.target_negative_fpr,
                "tau": threshold.tau,
                "sibling_alarm_rate": float(sib_fired),
                "clean_alarm_rate": float(clean_fired),
                "ratio": float(sib_fired / clean_fired) if clean_fired > 0 else np.inf,
            }
        )
    return pd.DataFrame(rows)


def analyze_probe(
    label: str, scores_path: Path, labels_path: Path, split: str, budgets: List[float]
) -> dict:
    frame = read_scores(scores_path, split=split)
    frame["rollout_score"] = [rollout_score(s, persistence=1) for s in frame["scores"]]
    loop_prone = prompt_predisposition(labels_path, split)

    siblings, clean = healthy_groups(frame, loop_prone)
    if siblings.empty or clean.empty:
        raise ValueError(
            f"{label}: one of the two groups is empty (siblings={len(siblings)}, "
            f"clean={len(clean)}) -- nothing to compare"
        )

    thresholds = choose_thresholds(frame, budgets, persistence=1, tuning_split=split)
    auc, p_value = rank_biserial_auc(
        siblings["rollout_score"].to_numpy(), clean["rollout_score"].to_numpy()
    )
    rates = alarm_rates(siblings, clean, thresholds)

    print(f"\n=== {label} ===")
    print(f"  healthy siblings of a loop-prone prompt: {len(siblings)}")
    print(f"  healthy rollouts of a clean prompt:       {len(clean)}")
    print(
        f"  sibling score  median={siblings['rollout_score'].median():.4f}  "
        f"mean={siblings['rollout_score'].mean():.4f}"
    )
    print(
        f"  clean score    median={clean['rollout_score'].median():.4f}  "
        f"mean={clean['rollout_score'].mean():.4f}"
    )
    print(
        f"  rank-biserial AUC (sibling > clean, healthy-only): {auc:.4f}  "
        f"(0.5 = no leakage)  Mann-Whitney p={p_value:.2e}"
    )
    print("\n  alarm rate at each frozen threshold:")
    print(rates.round(4).to_string(index=False))

    return {
        "label": label,
        "siblings": siblings,
        "clean": clean,
        "auc": auc,
        "p_value": p_value,
        "rates": rates,
    }


def plot_comparison(results: List[dict], out_dir: Path) -> Path:
    figure, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 4.0), sharey=True)
    if len(results) == 1:
        axes = [axes]

    for axis, result, colour in zip(axes, results, SERIES):
        for group_key, group_label, style in (
            ("clean", "clean prompt", dict(ls="-", lw=1.8)),
            ("siblings", "sibling of a loop-prone prompt", dict(ls=(0, (3, 2)), lw=1.8)),
        ):
            values = np.sort(result[group_key]["rollout_score"].to_numpy())
            fractions = np.arange(1, len(values) + 1) / len(values)
            axis.plot(values, fractions, color=colour, label=group_label, **style)
        axis.set_title(
            f"{result['label']}  (AUC={result['auc']:.3f})", color=INK, fontsize=11, loc="left"
        )
        axis.set_xlabel("rollout score (max over tokens)", color=INK_SOFT, fontsize=9)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(True, color=GRID, lw=0.6, alpha=0.9)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=INK_SOFT, labelsize=8)
        axis.legend(frameon=False, fontsize=8, loc="lower right")

    axes[0].set_ylabel("cumulative share of healthy rollouts", color=INK_SOFT, fontsize=9)
    figure.suptitle(
        "Healthy-rollout score, by whether the prompt has a degenerate sibling",
        color=INK, fontsize=12, x=0.008, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "prompt_predisposition_ecdf"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    return stem.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scores",
        nargs=2,
        action="append",
        metavar=("PATH", "LABEL"),
        required=True,
        help="A scores parquet (from score_rollouts.py) and a short label for it. Repeatable.",
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", default="val")
    parser.add_argument("--budgets", nargs="*", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "notebooks" / "figures" / "diary"
    )
    args = parser.parse_args()

    results = [
        analyze_probe(label, Path(path).resolve(), args.labels, args.split, args.budgets)
        for path, label in args.scores
    ]

    out_path = plot_comparison(results, args.output_dir)
    print(f"\nWrote {out_path} (and matching .pdf)")


if __name__ == "__main__":
    main()
