import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import re
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    from degeneration_probe.evaluation.head_selection import (
        StoppingRule,
        apply_rule_to_run,
    )
    from degeneration_probe.evaluation.protocol import WARNING_BANDS

    REPO = Path(__file__).resolve().parents[1]
    OUTPUTS = REPO / "outputs"
    RULE = StoppingRule()
    return (
        OUTPUTS,
        Path,
        RULE,
        StoppingRule,
        WARNING_BANDS,
        apply_rule_to_run,
        json,
        mo,
        np,
        pd,
        re,
    )


@app.cell
def _(OUTPUTS, Path, json, pd, re):
    def inventory():
        """Every run that has recorded a trajectory, with its recipe."""
        rows = []
        for info in OUTPUTS.glob("*/*/run_info.json"):
            if info.parent.name == "latest":
                continue
            trajectory = next(
                (
                    info.parent / name
                    for name in ("checkpoint_replay.parquet", "selection_history.parquet")
                    if (info.parent / name).is_file()
                ),
                None,
            )
            if trajectory is None:
                continue
            recorded = json.loads(info.read_text())
            config = json.loads((info.parent / "resolved_config.json").read_text())
            training = config["training"]
            selection, label, loss = (
                training["selection"],
                training["label"],
                training["loss"],
            )
            rows.append(
                {
                    "run": info.parent.parent.name,
                    "trajectory": trajectory,
                    "experiments": sorted(
                        set(re.findall(r"exp:(\w+)", ",".join(recorded.get("tags") or [])))
                    ),
                    "seed": training["runtime"]["seed"],
                    "regime": training["features"]["regime"],
                    "strategy": selection["strategy"],
                    "window": selection["window_size"],
                    "anchor": selection["anchor"],
                    "positive_fraction": selection["positive_fraction"],
                    "family": label["family"],
                    "horizon": label["horizon"],
                    "decay": label["decay"],
                    "decay_length": label["decay_length"],
                    "signal": label["signal"],
                    "loss": loss["name"],
                    "pos_weight": loss["bce"]["use_pos_weight"],
                    "build_root": config["dataset"]["build_root"],
                }
            )
        return pd.DataFrame(rows)

    def population(build_root, _cache={}):
        """How many answers of each kind an evaluation reads.

        Counted from the frontier labels rather than from a run's own summary,
        which counts answers by whatever target that run was trained on: under a
        regression target almost every answer holds a positive value, which says
        nothing about how many of them degenerate.
        """
        if build_root not in _cache:
            labels = pd.read_parquet(
                Path(build_root) / "onset_labels" / "onset_labels.parquet",
                columns=["split", "stop_reason", "onset_position", "is_positive"],
            )
            validation = labels[labels["split"] == "val"]
            degenerate = (
                validation["is_positive"].astype(bool)
                & validation["onset_position"].notna()
            ).sum()
            healthy = (validation["stop_reason"] == "eos").sum()
            _cache[build_root] = (int(degenerate), int(healthy))
        return _cache[build_root]

    RUNS = inventory()
    return RUNS, inventory, population


@app.cell
def _(RULE, RUNS, WARNING_BANDS, apply_rule_to_run, np, pd, population):
    def best_checkpoint(run):
        """The checkpoint the rule keeps, across every depth the run trained.

        Each depth is decided on its own, then the run is read at whichever depth
        reached the highest coverage of the run-up among the steps where its
        coverage inside the loop cleared the floor.
        """
        trajectory = pd.read_parquet(run.trajectory)
        outcomes = apply_rule_to_run(trajectory, RULE)
        selectable = outcomes[outcomes["selected_step"].notna()]
        if selectable.empty:
            return None
        pick = selectable.loc[selectable["selected_value"].idxmax()]
        record = trajectory[
            (trajectory["layer"] == pick["layer"])
            & (trajectory["step"] == pick["selected_step"])
        ].iloc[0]
        # Precision at the frozen operating point. The record carries the share
        # of healthy answers that fired and the share of degenerate ones caught,
        # and the split says how many there are of each, which is everything a
        # confusion matrix needs.
        degenerate, healthy = population(run.build_root)
        caught = record["recall_at_budget"] * degenerate
        false_alarms = record["budget_realized_fpr"] * healthy
        return {
            "layer": int(pick["layer"]),
            "step": int(pick["selected_step"]),
            "rollout recall": record["recall_at_budget"],
            "rollout precision": caught / (caught + false_alarms)
            if caught + false_alarms
            else np.nan,
            # What the threshold actually spent. Precision is pinned by this
            # rather than by the probe, and a scorer whose healthy answers pile
            # up on one value cannot spend the budget at all, which is the case
            # where every other number changes meaning.
            "false-alarm rate": record["budget_realized_fpr"],
            "in-pattern recall": record["in_pattern_recall"],
            **{
                f"warning coverage {band}": record.get(f"warning_recall_{band}", np.nan)
                for band in WARNING_BANDS
            },
            "median offset": record["median_offset"],
            # The median is taken over the answers that fired, so a probe that
            # misses the hard cases gets a flattering one. This is the count it
            # has to be read against.
            "never fired": int(record["never_fired_positives"]),
        }

    def described(run):
        """The recipe, named by the settings that distinguish it."""
        parts = [run.strategy]
        if run.strategy != "all_tokens":
            parts.append(f"W={run.window}")
        if run.strategy in {"frontier_window", "frontier_window_hard_negative", "random_window"}:
            parts.append(run.anchor)
        if run.family == "frontier_hard":
            parts.append(f"hard h={run.horizon}")
        elif run.family == "frontier_soft":
            parts.append(f"soft {run.decay} {int(run.decay_length)}")
        else:
            parts.append(run.signal)
        parts.append(run.loss)
        if not run.pos_weight:
            parts.append("pos_weight off")
        if run.positive_fraction != 0.25:
            parts.append(f"positive fraction {run.positive_fraction}")
        if run.regime == "adapted":
            parts.append("adapters")
        return " ".join(str(part) for part in parts)

    def scored_runs():
        rows = []
        for run in RUNS.itertuples():
            best = best_checkpoint(run)
            if best is None:
                continue
            rows.append(
                {
                    "recipe": described(run),
                    "seed": run.seed,
                    "experiments": run.experiments,
                    **best,
                }
            )
        return pd.DataFrame(rows)

    SCORED = scored_runs()
    return SCORED, best_checkpoint, described, scored_runs


@app.cell
def _(SCORED, WARNING_BANDS, mo, pd):
    OBJECTIVE = "warning coverage 256"
    # Averaged over a recipe's seeds, each to the precision it is read at, so a
    # figure and its spread are written to the same number of places.
    AVERAGED = {
        "rollout recall": 3,
        "rollout precision": 3,
        "false-alarm rate": 4,
        "in-pattern recall": 3,
        **{f"warning coverage {band}": 4 for band in WARNING_BANDS},
        "median offset": 0,
        "never fired": 1,
    }
    # A depth and a step are each seed's own pick out of a discrete set, so the
    # middle one is reported with the range the seeds covered. A standard
    # deviation would describe a spread around a mean that names no checkpoint
    # any seed selected.
    RANGED = ["layer", "step"]
    COLUMNS = [
        "training strategy",
        "rank",
        "seeds",
        *RANGED,
        *AVERAGED,
    ]

    def with_spread(mean, spread, decimals):
        """A figure and how far its seeds sat from it.

        Written into the one cell rather than into a column of its own, because
        a spread parked at the far end of a wide table is read as a separate
        fact when it is the error bar on the number beside the rank.
        """
        if pd.isna(mean):
            return ""
        if pd.isna(spread):
            return f"{mean:.{decimals}f}"
        return f"{mean:.{decimals}f} ± {spread:.{decimals}f}"

    def with_range(middle, low, high):
        """The middle pick, and the range the seeds actually covered."""
        if pd.isna(middle):
            return ""
        if low == high:
            return f"{int(middle)}"
        return f"{int(middle)} ({int(low)}-{int(high)})"

    def table(stages, title):
        """One row per training strategy, averaged over its seeds.

        Each seed contributes the checkpoint the rule kept for it, so a row is
        the mean of three separately selected probes rather than of three
        readings of one, and every figure carries the spread of those seeds. That
        spread is what says whether the ordering above is worth reading: repeats
        of one recipe differ by more than neighbouring rows in these tables do.
        """
        if SCORED.empty:
            return mo.md(f"**{title}** — no run has recorded a trajectory yet.")
        wanted = set(stages)
        rows = SCORED[SCORED["experiments"].map(lambda tags: bool(wanted & set(tags)))]
        if rows.empty:
            return mo.md(f"**{title}** — no run has finished yet.")
        grouped = rows.groupby("recipe")
        columns = list(AVERAGED)
        means = grouped[columns].mean()
        spreads = grouped[columns].std()
        order = means.sort_values(OBJECTIVE, ascending=False).index

        shown = pd.DataFrame(index=order)
        shown["training strategy"] = order
        shown["rank"] = range(1, len(order) + 1)
        shown["seeds"] = grouped.size().reindex(order)
        for name in RANGED:
            middles = grouped[name].median().reindex(order)
            lows = grouped[name].min().reindex(order)
            highs = grouped[name].max().reindex(order)
            shown[name] = [
                with_range(middles[recipe], lows[recipe], highs[recipe]) for recipe in order
            ]
        for name, decimals in AVERAGED.items():
            shown[name] = [
                with_spread(means.loc[recipe, name], spreads.loc[recipe, name], decimals)
                for recipe in order
            ]
        return mo.vstack(
            [mo.md(f"### {title}"), mo.ui.table(shown[COLUMNS].reset_index(drop=True), selection=None)]
        )

    return AVERAGED, COLUMNS, OBJECTIVE, RANGED, table


@app.cell
def _(mo, table):
    mo.vstack(
        [
            table(["S1"], "S1. Token selection"),
            table(["S2a", "S2b", "S2c", "S2d"], "S2. Label selection"),
            table(["S3"], "S3. Adapters"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
