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

    from degeneration_probe.evaluation.protocol import persistence_scores

    REPO = Path("/iopsstor/scratch/cscs/mdenegri/degeneration-probe")
    OUTPUTS = REPO / "outputs"
    FIGURES = REPO / "notebooks" / "figures" / "difficulty"
    FIGURES.mkdir(parents=True, exist_ok=True)

    # The reference categorical palette, in fixed slot order. A rule keeps its
    # colour wherever it appears, so a filter that drops rungs never repaints
    # the survivors.
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
    COLOUR = {key: SERIES[i] for i, (key, _) in enumerate(RUNGS)}
    LABEL = dict(RUNGS)

    def save_figure(figure, name):
        """Every figure leaves a PDF beside it, for the write-up."""
        figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
        figure.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
        return figure

    def style(axis, title, xlabel, ylabel):
        axis.set_title(title, color=INK, fontsize=11, loc="left")
        axis.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
        axis.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)
        axis.grid(True, color=GRID, lw=0.6, alpha=0.9)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=INK_SOFT, labelsize=8)
        return axis

    return (
        COLOUR,
        INK_SOFT,
        LABEL,
        OUTPUTS,
        Path,
        RUNGS,
        json,
        mo,
        np,
        pd,
        persistence_scores,
        plt,
        save_figure,
        style,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # What each probe finds easy

    Two probes can catch the same number of degenerate rollouts and still be
    reading different things. This notebook asks where in a rollout each
    selection rule actually finds its evidence, and whether the rollouts one
    rule finds easy are the rollouts another finds easy.

    The expectation worth testing is that degeneration gets easier to spot the
    further into the loop you are, since the text is by then obviously broken.
    If that holds for every rule, the rules differ only in sensitivity. If it
    holds for some and not others, they have learned different concepts, and the
    choice between them is not a matter of tuning.

    ## What anchors the position axis, and what does not

    The only landmark available is the judge's onset, which marks where the first
    loop begins. It does not mark where the loop *ends*, and no trustworthy
    measure of the loop's extent exists yet, so "depth into the loop" here means
    distance from that onset and nothing more. A rollout whose first loop is
    short and one whose first loop runs to the end of the generation are treated
    alike, which is a real limitation of every position-conditioned number below.

    The onset is also a statement about the text rather than about the model. The
    representation may well be degenerate before the text visibly is, so a small
    positive offset is not automatically a failure to anticipate.
    """)
    return


@app.cell
def _(OUTPUTS, Path, json):
    def find_ladder_runs():
        """Every equal-budget ladder run that has been scored, by rule and seed."""
        found = {}
        for config_path in sorted(OUTPUTS.glob("*/*/resolved_config.json")):
            if config_path.parent.is_symlink():
                continue
            config = json.loads(config_path.read_text())
            training = config["training"]
            selection = training.get("selection")
            if not selection or training["probe"].get("layers") is None:
                continue
            if training["runtime"]["max_steps"] != 800:
                continue
            if selection["window_size"] != 128 or selection["anchor"] != "centered":
                continue
            if training["label"]["family"] != "frontier_hard":
                continue
            key = (selection["strategy"], training["runtime"]["seed"])
            found[key] = config_path.parent
        return found

    LADDER = find_ladder_runs()

    def scored_depths(run_dir: Path):
        return sorted(
            int(p.name.split("_")[-1])
            for p in run_dir.glob("layers/layer_*")
            if (p / "scores" / "val.parquet").is_file()
        )

    DEPTHS = sorted({d for run in LADDER.values() for d in scored_depths(run)})
    SEEDS = sorted({seed for _, seed in LADDER})
    return DEPTHS, LADDER, SEEDS


@app.cell
def _(DEPTHS, LADDER, SEEDS, mo):
    mo.stop(
        not LADDER,
        mo.md(
            "**No scored ladder run found.** Train the five rungs at an equal "
            "budget, then score a depth of each with `cluster/score_layers.sbatch`."
        ).callout(kind="warn"),
    )

    layer_pick = mo.ui.dropdown(
        options={str(d): d for d in DEPTHS},
        value=str(12 if 12 in DEPTHS else DEPTHS[0]),
        label="Depth",
    )
    seed_pick = mo.ui.dropdown(
        options={str(s): s for s in SEEDS}, value=str(SEEDS[0]), label="Seed"
    )
    budget_pick = mo.ui.dropdown(
        options={"1%": 0.01, "5%": 0.05, "10%": 0.10}, value="10%", label="False-alarm budget"
    )
    persistence_pick = mo.ui.dropdown(
        options={str(m): m for m in (1, 2, 4, 8, 16, 32)},
        value="4",
        label="Persistence (consecutive tokens)",
    )
    mo.hstack([layer_pick, seed_pick, budget_pick, persistence_pick], justify="start", gap=1.5)
    return budget_pick, layer_pick, persistence_pick, seed_pick


@app.cell
def _(
    LADDER,
    RUNGS,
    budget_pick,
    json,
    layer_pick,
    np,
    pd,
    persistence_pick,
    persistence_scores,
    seed_pick,
):
    def load_rung(strategy):
        """One rule's scored rollouts at the chosen depth, with its own threshold.

        The threshold is the rule's own, re-derived per run, because a rule that
        scores everything a little higher is not thereby better and a shared
        threshold would say it was.
        """
        run_dir = LADDER.get((strategy, seed_pick.value))
        if run_dir is None:
            return None
        base = run_dir / "layers" / f"layer_{layer_pick.value:02d}"
        if not (base / "scores" / "val.parquet").is_file():
            return None
        frame = pd.read_parquet(base / "scores" / "val.parquet")
        frozen = json.loads((base / "decision_thresholds.json").read_text())
        wanted = min(
            frozen["thresholds"],
            key=lambda t: abs(t["target_negative_fpr"] - budget_pick.value),
        )
        return frame, float(wanted["tau"])

    def held(scores):
        """The score after requiring the chosen run of consecutive tokens."""
        return persistence_scores(np.asarray(scores, dtype=np.float32), persistence_pick.value)

    LOADED = {}
    for _strategy, _ in RUNGS:
        _got = load_rung(_strategy)
        if _got is not None:
            LOADED[_strategy] = _got

    RULES = [s for s, _ in RUNGS if s in LOADED]
    return LOADED, RULES, held


@app.cell
def _(LOADED, RULES, held, np, pd):
    def per_rollout():
        """One row per rule and degenerate rollout: where it fired and how hard.

        `offset` is the first alarm relative to the judge's onset, so a negative
        value means the probe committed before the loop was visible in the text.
        """
        rows = []
        for rule in RULES:
            frame, tau = LOADED[rule]
            positives = frame[frame.is_positive]
            for row in positives.itertuples():
                scores = np.asarray(row.scores, dtype=np.float32)
                onset = int(row.onset_position)
                above = held(scores) >= tau
                fired = np.flatnonzero(above)
                in_pattern = above[onset:]
                rows.append(
                    {
                        "rule": rule,
                        "prompt_id": row.prompt_id,
                        "rollout_idx": row.rollout_idx,
                        "domain": row.domain,
                        "onset": onset,
                        "num_tokens": row.num_tokens,
                        "loop_tokens": len(scores) - onset,
                        "caught": bool(fired.size),
                        "offset": int(fired[0]) - onset if fired.size else np.nan,
                        "coverage": float(in_pattern.mean()) if in_pattern.size else np.nan,
                        "peak": float(scores.max()),
                    }
                )
        return pd.DataFrame(rows)

    ROLLOUTS = per_rollout()
    ROLLOUTS.groupby("rule").agg(
        rollouts=("caught", "size"),
        caught=("caught", "sum"),
        median_offset=("offset", "median"),
        mean_coverage=("coverage", "mean"),
    ).round(3)
    return (ROLLOUTS,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Does detection get easier deeper into the loop?

    For every degenerate rollout, at each distance from the onset, the share of
    rollouts whose probe is firing at that position. A curve that climbs means
    the rule finds the loop progressively more obvious. A curve that is already
    high at the onset and stays flat means the rule decided at the boundary and
    learned nothing further from the loop itself.
    """)
    return


@app.cell
def _(LOADED, RULES, held, np):
    def firing_profile(span=(-256, 1024)):
        """Share of degenerate rollouts firing, by distance from the onset."""
        low, high = span
        offsets = np.arange(low, high)
        profile = {}
        for rule in RULES:
            frame, tau = LOADED[rule]
            total = np.zeros(high - low)
            counts = np.zeros(high - low)
            for row in frame[frame.is_positive].itertuples():
                scores = np.asarray(row.scores, dtype=np.float32)
                onset = int(row.onset_position)
                above = held(scores) >= tau
                start = max(low, -onset)
                stop = min(high, len(scores) - onset)
                if stop <= start:
                    continue
                index = slice(start - low, stop - low)
                total[index] += above[onset + start : onset + stop]
                counts[index] += 1
            profile[rule] = (total / np.where(counts == 0, np.nan, counts), counts)
        return offsets, profile

    PROFILE_OFFSETS, PROFILE = firing_profile()
    return PROFILE, PROFILE_OFFSETS


@app.cell
def _(
    COLOUR,
    INK_SOFT,
    LABEL,
    PROFILE,
    PROFILE_OFFSETS,
    RULES,
    plt,
    save_figure,
    style,
):
    _fig, _ax = plt.subplots(figsize=(7.6, 4.4))
    for _rule in RULES:
        _share, _counts = PROFILE[_rule]
        _thin = _counts < 20
        _plot = _share.copy()
        _plot[_thin] = float("nan")
        _ax.plot(PROFILE_OFFSETS, _plot, color=COLOUR[_rule], lw=1.8, label=LABEL[_rule])
    _ax.axvline(0, color=INK_SOFT, lw=1, ls=(0, (4, 3)))
    _ax.annotate(
        "onset",
        xy=(0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -10),
        textcoords="offset points",
        color=INK_SOFT,
        fontsize=8,
    )
    style(
        _ax,
        "Share of degenerate rollouts firing, by distance from the onset",
        "tokens from the onset (negative is before)",
        "share firing",
    )
    _ax.set_ylim(0, 1)
    _legend = _ax.legend(frameon=False, fontsize=8, loc="lower right")
    for _text in _legend.get_texts():
        _text.set_color(INK_SOFT)
    save_figure(_fig, "firing_profile_absolute")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The same question without the length confound

    A rollout whose loop runs for two thousand tokens and one whose loop runs for
    two hundred contribute to completely different parts of the axis above, and
    the long ones are the only ones left at the right-hand edge. Rescaling each
    rollout's loop to run from zero to one removes that, at the cost of making
    the axis a proportion rather than a count.
    """)
    return


@app.cell
def _(LOADED, RULES, held, np):
    def scaled_profile(bins=50):
        """Firing share against the fraction of the way through each loop."""
        edges = np.linspace(0.0, 1.0, bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        profile = {}
        for rule in RULES:
            frame, tau = LOADED[rule]
            total = np.zeros(bins)
            counts = np.zeros(bins)
            for row in frame[frame.is_positive].itertuples():
                scores = np.asarray(row.scores, dtype=np.float32)
                onset = int(row.onset_position)
                tail = held(scores)[onset:] >= tau
                if tail.size < bins:
                    continue
                position = np.linspace(0.0, 1.0, tail.size)
                which = np.clip(np.searchsorted(edges, position, side="right") - 1, 0, bins - 1)
                total += np.bincount(which, weights=tail, minlength=bins)
                counts += np.bincount(which, minlength=bins)
            profile[rule] = total / np.where(counts == 0, np.nan, counts)
        return centres, profile

    SCALED_CENTRES, SCALED = scaled_profile()
    return SCALED, SCALED_CENTRES


@app.cell
def _(COLOUR, LABEL, RULES, SCALED, SCALED_CENTRES, plt, save_figure, style):
    _fig, _ax = plt.subplots(figsize=(7.6, 4.4))
    for _rule in RULES:
        _ax.plot(SCALED_CENTRES, SCALED[_rule], color=COLOUR[_rule], lw=1.8, label=LABEL[_rule])
    style(
        _ax,
        "Firing share against position within each rollout's own loop",
        "fraction of the way from the onset to the end of the rollout",
        "share of tokens firing",
    )
    _ax.set_ylim(0, 1)
    _legend = _ax.legend(frameon=False, fontsize=8, loc="lower right")
    for _text in _legend.get_texts():
        _text.set_color("#52514e")
    save_figure(_fig, "firing_profile_scaled")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Do the rules find the same rollouts easy?

    Two rules can reach the same coverage by succeeding on the same rollouts or
    on different ones, and only the second case means they are complementary.
    Per rollout, the first-alarm offset under one rule against the other: points
    on the diagonal are rollouts both rules read the same way.
    """)
    return


@app.cell
def _(ROLLOUTS, pd):
    WIDE = ROLLOUTS.pivot_table(
        index=["prompt_id", "rollout_idx", "domain"], columns="rule", values="offset"
    )
    AGREEMENT = WIDE.corr(method="spearman").round(3)
    pd.concat(
        {
            "rank agreement of first-alarm offset": AGREEMENT,
        },
        axis=0,
    )
    return (WIDE,)


@app.cell
def _(COLOUR, LABEL, RULES, WIDE, np, plt, save_figure, style):
    def difficulty_scatter(left, right):
        _fig, _ax = plt.subplots(figsize=(5.4, 5.2))
        _both = WIDE[[left, right]].dropna()
        _ax.scatter(
            _both[left],
            _both[right],
            s=22,
            color=COLOUR[right],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.85,
        )
        _lo = float(np.nanmin(_both.to_numpy()))
        _hi = float(np.nanmax(_both.to_numpy()))
        _ax.plot([_lo, _hi], [_lo, _hi], color="#52514e", lw=1, ls=(0, (4, 3)))
        _ax.axhline(0, color="#d8d7d2", lw=1)
        _ax.axvline(0, color="#d8d7d2", lw=1)
        style(
            _ax,
            f"First-alarm offset, {LABEL[left]} against {LABEL[right]}",
            f"{LABEL[left]} (tokens from onset)",
            f"{LABEL[right]} (tokens from onset)",
        )
        save_figure(_fig, f"difficulty_{left}_vs_{right}")
        return _fig

    _pair = ("all_tokens", "frontier_window")
    difficulty_scatter(*_pair) if all(p in RULES for p in _pair) else None
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Which rollouts nobody catches

    A rollout missed by every rule is a property of the corpus or of the label,
    not of the selection rule. A rollout missed by one rule alone is the
    interesting kind, since something about how that rule was trained made this
    example invisible to it.
    """)
    return


@app.cell
def _(ROLLOUTS, RULES, pd):
    def miss_overlap():
        missed = {
            rule: set(
                map(
                    tuple,
                    ROLLOUTS[(ROLLOUTS.rule == rule) & (~ROLLOUTS.caught)][
                        ["prompt_id", "rollout_idx"]
                    ].to_numpy(),
                )
            )
            for rule in RULES
        }
        everyone = set.intersection(*missed.values()) if missed else set()
        rows = [
            {
                "rule": rule,
                "missed": len(missed[rule]),
                "missed by this rule alone": len(
                    missed[rule] - set.union(*[m for r, m in missed.items() if r != rule])
                ),
            }
            for rule in RULES
        ]
        return pd.DataFrame(rows), everyone

    MISS_TABLE, MISSED_BY_ALL = miss_overlap()
    MISS_TABLE.assign(missed_by_every_rule=len(MISSED_BY_ALL))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Per domain

    Domains differ in how much legitimate repetition they contain, so a rule that
    looks better overall may simply be better on whichever domain contributes
    most of the positives. Any cell backed by very few positive rollouts is
    anecdotal and is marked as such rather than read as an estimate.
    """)
    return


@app.cell
def _(ROLLOUTS):
    BY_DOMAIN = (
        ROLLOUTS.groupby(["domain", "rule"])
        .agg(
            positives=("caught", "size"),
            caught=("caught", "sum"),
            median_offset=("offset", "median"),
            mean_coverage=("coverage", "mean"),
        )
        .round(3)
        .reset_index()
    )
    BY_DOMAIN.assign(anecdotal=BY_DOMAIN["positives"] < 10)
    return (BY_DOMAIN,)


@app.cell
def _(BY_DOMAIN, COLOUR, LABEL, RULES, np, plt, save_figure, style):
    _domains = sorted(BY_DOMAIN["domain"].unique())
    _fig, _ax = plt.subplots(figsize=(8.4, 4.4))
    _width = 0.8 / max(1, len(RULES))
    _x = np.arange(len(_domains))
    for _i, _rule in enumerate(RULES):
        _sub = BY_DOMAIN[BY_DOMAIN.rule == _rule].set_index("domain")
        _vals = [_sub["mean_coverage"].get(d, np.nan) for d in _domains]
        _ax.bar(
            _x + _i * _width - 0.4 + _width / 2,
            _vals,
            width=_width * 0.9,
            color=COLOUR[_rule],
            label=LABEL[_rule],
        )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_domains, rotation=20, ha="right")
    style(_ax, "Token coverage inside the loop, by domain", "", "mean coverage")
    _ax.set_ylim(0, 1)
    _legend = _ax.legend(frameon=False, fontsize=8, ncol=2)
    for _text in _legend.get_texts():
        _text.set_color("#52514e")
    save_figure(_fig, "coverage_by_domain")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notes for whoever picks this up

    - Scores come from `<run>/layers/layer_NN/scores/val.parquet`, written by
      `cluster/score_layers.sbatch`. Each depth also carries its own frozen
      thresholds beside them, and those are the thresholds used here.
    - Thresholds are per rule. A rule whose scores sit higher everywhere is not
      thereby better, and a shared threshold would report that it was.
    - The persistence control re-uses the same rule the protocol applies, so a
      value of one reproduces the reported numbers exactly. It does **not**
      re-derive the threshold for the chosen persistence, unlike the protocol's
      own sweep, so at large values the budget shown on the control is no longer
      the budget being spent.
    - The longest-repeated-substring signal is stored in the corpus and is not
      used anywhere here. It does not locate a loop reliably enough to anchor a
      position axis.
    """)
    return


if __name__ == "__main__":
    app.run()
