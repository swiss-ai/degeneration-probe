import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    REPO = Path("/iopsstor/scratch/cscs/mdenegri/degeneration-probe")
    OUTPUTS = REPO / "outputs"
    FIGURES = REPO / "notebooks" / "figures" / "runs"
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES, OUTPUTS, Path, json, mo, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Runs and what they detect

    Two halves. The first reads what training left behind: every attempt of
    every recipe, its axes, and its curves. The second reads the evaluation
    protocol's output for whichever scorers have been through it, probes and
    heuristic baselines alike, since both are judged the same way.

    Nothing here recomputes anything. It reads the run directories, which is
    why they were made self-describing.
    """)
    return


@app.cell
def _(OUTPUTS, json, pd):
    def load_runs(root):
        rows = []
        for info_path in sorted(root.glob("*/*/run_info.json")):
            if info_path.parent.is_symlink():
                continue
            try:
                info = json.loads(info_path.read_text())
            except ValueError:
                continue
            training = info.get("training", {})
            rows.append(
                {
                    "run_name": info.get("run_name"),
                    "attempt": info.get("attempt"),
                    "group": info.get("group"),
                    "status": info.get("status"),
                    "minutes": round((info.get("duration_seconds") or 0) / 60, 1),
                    "steps": training.get("global_step"),
                    "best_metric": training.get("best_metric"),
                    "selected_on": training.get("metric_for_best_model"),
                    "commit": (info.get("environment", {}).get("git_commit") or "")[:8],
                    "dirty": info.get("environment", {}).get("git_dirty"),
                    "wandb": (info.get("wandb") or {}).get("url"),
                    "run_dir": str(info_path.parent),
                    **{f"axis.{k}": v for k, v in (info.get("axes") or {}).items()},
                }
            )
        return pd.DataFrame(rows)

    runs = load_runs(OUTPUTS)
    runs
    return (runs,)


@app.cell
def _(mo, runs):
    finished = runs[runs["status"] == "finished"] if len(runs) else runs
    run_picker = mo.ui.multiselect(
        options=sorted(finished["run_dir"].tolist()) if len(finished) else [],
        value=sorted(finished["run_dir"].tolist())[:4] if len(finished) else [],
        label="Runs to compare",
    )
    run_picker
    return (run_picker,)


@app.cell
def _(Path, pd, run_picker):
    def load_history(run_dirs):
        frames = []
        for run_dir in run_dirs:
            path = Path(run_dir) / "history.parquet"
            if not path.is_file():
                continue
            frame = pd.read_parquet(path)
            frame["run"] = Path(run_dir).parent.name
            frame["attempt"] = Path(run_dir).name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    history = load_history(run_picker.value)
    history
    return (history,)


@app.cell
def _(mo):
    mo.md("""
    ## Training curves

    The monitoring curve and the diagnostics that ride with it. The score
    spread is the one to watch alongside the loss: a probe that collapses to
    a constant keeps a plausible loss while distinguishing nothing.
    """)
    return


@app.cell
def _(FIGURES, history, plt):
    def plot_curves(frame, path):
        columns = [
            ("loss", "training loss"),
            ("val/loss", "validation loss"),
            ("val/rollout_auc", "rollout AUC"),
            ("val/prediction_std", "score spread"),
        ]
        available = [(c, title) for c, title in columns if c in frame.columns]
        fig, axes = plt.subplots(
            1, len(available), figsize=(4.2 * len(available), 3.6), squeeze=False
        )
        for axis, (column, title) in zip(axes[0], available):
            for run, group in frame.groupby("run"):
                points = group[["step", column]].dropna()
                if len(points):
                    axis.plot(points["step"], points[column], marker="o", ms=3, label=run[:34])
            if column == "val/prediction_std":
                axis.axhline(0.01, color="crimson", ls="--", lw=1)
            axis.set_title(title)
            axis.set_xlabel("step")
            axis.grid(alpha=0.3)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        return fig

    curves = (
        plot_curves(history, FIGURES / "training_curves")
        if len(history)
        else None
    )
    curves
    return


@app.cell
def _(mo):
    mo.md("""
    ## What the probe sees

    The composition of each split as the run actually saw it, and the class
    weight beside it. The effective ratio lands on one when the weight
    matches the sampled population, which is what says the imbalance was
    corrected once rather than twice.
    """)
    return


@app.cell
def _(Path, json, pd, run_picker):
    def load_composition(run_dirs):
        rows = []
        for run_dir in run_dirs:
            path = Path(run_dir) / "dataset_summary.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text())
            for split, values in summary.items():
                if not isinstance(values, dict):
                    continue
                rows.append(
                    {
                        "run": Path(run_dir).parent.name[:44],
                        "split": split,
                        "rollouts": values.get("rollouts") or values.get("windows"),
                        "positive_rollouts": values.get("positive_rollouts"),
                        "valid_tokens": values.get("valid_tokens"),
                        "positive_token_rate": round(values.get("positive_token_rate", 0), 4),
                    }
                )
            weight = summary.get("pos_weight")
            for row in rows[-len([k for k, v in summary.items() if isinstance(v, dict)]) :]:
                row["pos_weight"] = round(weight, 4) if isinstance(weight, (int, float)) else None
                row["effective_positive_ratio"] = summary.get("effective_positive_ratio")
        return pd.DataFrame(rows)

    composition = load_composition(run_picker.value)
    composition
    return


@app.cell
def _(mo):
    mo.md("""
    ## The evaluation protocol

    Any scorer that has been through `evaluate_scores` appears here, probe
    or baseline. The four views are reported together because none is
    interpretable alone: a scorer that fires on every token has perfect
    recall and perfect persistence, and only its false-alarm rate says so.
    """)
    return


@app.cell
def _(OUTPUTS, mo):
    scored = sorted(
        str(path.parent.parent.parent)
        for path in OUTPUTS.glob("**/evaluation/*/view_a_detection.csv")
    )
    scorer_picker = mo.ui.multiselect(
        options=sorted(set(scored)),
        value=sorted(set(scored)),
        label="Scorers",
    )
    split_picker = mo.ui.dropdown(
        options=["val", "test_indomain", "test_heldout_domains"],
        value="val",
        label="Split",
    )
    mo.hstack([scorer_picker, split_picker])
    return scorer_picker, split_picker


@app.cell
def _(Path, pd, scorer_picker, split_picker):
    def load_view(scorers, split, view):
        frames = []
        for scorer in scorers:
            path = Path(scorer) / "evaluation" / split / f"{view}.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "scorer", Path(scorer).name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    detection = load_view(scorer_picker.value, split_picker.value, "view_a_detection")
    detection
    return detection, load_view


@app.cell
def _(load_view, scorer_picker, split_picker):
    lead_time = load_view(scorer_picker.value, split_picker.value, "view_c_lead_time")
    lead_time
    return (lead_time,)


@app.cell
def _(load_view, scorer_picker, split_picker):
    persistence = load_view(scorer_picker.value, split_picker.value, "view_d_persistence")
    persistence
    return (persistence,)


@app.cell
def _(mo):
    mo.md("""
    ### Detection against its cost

    Recall on one axis, the false-alarm rate that bought it on the other.
    A point is one scorer at one frozen operating point.
    """)
    return


@app.cell
def _(FIGURES, detection, lead_time, plt):
    def plot_operating_points(detection_frame, lead_frame, path):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for scorer, group in detection_frame.groupby("scorer"):
            group = group.sort_values("negative_fpr")
            axes[0].plot(
                group["negative_fpr"], group["recall"], marker="o", label=scorer[:30]
            )
            axes[1].plot(
                group["negative_fpr"], group["precision"], marker="o", label=scorer[:30]
            )
        axes[0].set_xlabel("false alarms per negative rollout")
        axes[0].set_ylabel("recall")
        axes[0].set_title("what recall costs")
        axes[1].set_xlabel("false alarms per negative rollout")
        axes[1].set_ylabel("precision")
        axes[1].set_title("precision at the same points")
        for axis in axes:
            axis.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        return fig

    operating_points = (
        plot_operating_points(detection, lead_time, FIGURES / "operating_points")
        if len(detection)
        else None
    )
    operating_points
    return


@app.cell
def _(mo):
    mo.md("""
    ### Lead time, and whether the alarm holds

    Negative lead time means the scorer fired before the frontier. The
    persistence panel is read in opposite directions for the two
    populations: on positives a sticky alarm is the goal, while on negatives
    every alarm is an error and the run length separates a jittery scorer
    from a confidently wrong one.
    """)
    return


@app.cell
def _(FIGURES, lead_time, persistence, plt):
    def plot_lead_and_persistence(lead_frame, persistence_frame, path):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for scorer, group in lead_frame.groupby("scorer"):
            group = group.sort_values("target_negative_fpr")
            axes[0].plot(
                group["target_negative_fpr"], group["median_offset"], marker="o", label=scorer[:30]
            )
            axes[1].plot(
                group["target_negative_fpr"],
                group["never_fired_positives"],
                marker="o",
                label=scorer[:30],
            )
        axes[0].axhline(0, color="black", lw=1)
        axes[0].set_ylabel("median tokens relative to the frontier")
        axes[0].set_title("lead time (below zero is early)")
        axes[1].set_ylabel("positives never caught")
        axes[1].set_title("misses")
        for scorer, group in persistence_frame.groupby("scorer"):
            for population, marker in (("positive", "o"), ("negative", "x")):
                subset = group[group["population"] == population].sort_values(
                    "target_negative_fpr"
                )
                if len(subset):
                    axes[2].plot(
                        subset["target_negative_fpr"],
                        subset["median_first_run_length"],
                        marker=marker,
                        ls="-" if population == "positive" else "--",
                        label=f"{scorer[:22]} ({population})",
                    )
        axes[2].set_yscale("log")
        axes[2].set_ylabel("median first run length (tokens)")
        axes[2].set_title("how long an alarm holds")
        for axis in axes:
            axis.set_xlabel("false-alarm budget")
            axis.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        return fig

    lead_and_persistence = (
        plot_lead_and_persistence(lead_time, persistence, FIGURES / "lead_and_persistence")
        if len(lead_time) and len(persistence)
        else None
    )
    lead_and_persistence
    return


@app.cell
def _(mo):
    mo.md("""
    ### Per-domain, for held-out splits

    Never pooled. The held-out domains differ sharply in how often they
    degenerate, and cells marked anecdotal are backed by too few positives
    to estimate anything.
    """)
    return


@app.cell
def _(load_view, scorer_picker, split_picker):
    per_domain = load_view(scorer_picker.value, split_picker.value, "per_domain_detection")
    per_domain
    return


if __name__ == "__main__":
    app.run()
