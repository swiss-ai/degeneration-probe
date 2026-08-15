import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import shlex
    from pathlib import Path

    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    REPO = Path("/iopsstor/scratch/cscs/mdenegri/degeneration-probe")
    OUTPUTS = REPO / "outputs"
    BASELINE_ROOT = OUTPUTS / "baselines"
    BUILD_ROOT = Path(
        "/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe"
        "/degeneration-dataset-apertus-8b-instruct"
    )
    FIGURES = REPO / "notebooks" / "figures" / "diary"
    FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        BASELINE_ROOT,
        BUILD_ROOT,
        FIGURES,
        OUTPUTS,
        Path,
        REPO,
        json,
        mo,
        mpl,
        np,
        pd,
        plt,
        shlex,
    )


@app.cell
def _(mo):
    mo.md("""
    # Detecting degeneration before it happens

    Language models sometimes fall into a loop. Partway through an answer they
    begin repeating a phrase and never stop, until the token limit cuts them
    off. This project asks whether that failure is visible **inside the model's
    own activations, while it is happening**, early enough that acting on the
    warning would be worth something.

    The answer has to be more than "a classifier can tell a broken answer from a
    good one". That is easy, because a broken answer is mostly loop by the time
    it ends. The question that matters is whether anything shows up in the
    *approach*, in the tokens just before the text visibly breaks.

    This notebook is the running record: what the system is, what each
    experiment asks, exactly which runs were trained, and what came back. Every
    number is read from a run directory on disk. Sections whose runs are not
    finished say so rather than showing a stale answer.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## How the pieces fit together

    **A corpus of answers.** Roughly 36,000 answers generated from an 8B model.
    An answer that finished on its own is a **negative**. One that was still
    looping when it hit the token limit is a **positive**.

    **A ground-truth frontier.** Every positive answer has one token position
    where the degeneration begins, the **frontier**. An LLM judge reads the
    answer and marks the spot by quoting the text; a second step finds that
    quote in the token stream. Asking for a quote rather than a number is what
    makes the label checkable, both automatically and by a person reading it.

    **A probe.** A linear head reading the residual stream at one layer,
    producing a score between 0 and 1 for every token. About twelve thousand
    parameters. Because the cost of a run is dominated by reading activations
    off disk, one run carries a separate head at every layer for barely more
    than the price of one, so depth is something every result already has rather
    than a separate sweep.

    **A rule for which tokens to train on.** An answer has thousands of tokens
    and almost all of them say nothing. Five rules form a ladder, from "use
    every token" up to "use a window around the frontier, and pick negative
    windows that already look repetitive". Every rule gets the same number of
    tokens per optimizer step, so they can be compared without compute being the
    difference between them.

    **An evaluation protocol that never sees the probe.** A run writes one score
    per token to a file, and the evaluator reads only that file. Probes and
    simple heuristics therefore go through exactly the same judgement.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## The corpus
    """)
    return


@app.cell
def _(BUILD_ROOT, mo, pd):
    def corpus_tables():
        labels = pd.read_parquet(BUILD_ROOT / "onset_labels" / "onset_labels.parquet")

        capped = labels[labels["stop_reason"] != "eos"]
        resolution = (
            capped["onset_resolution"]
            .fillna("unresolved")
            .value_counts()
            .rename_axis("outcome")
            .reset_index(name="rollouts")
        )
        resolution["outcome"] = resolution["outcome"].replace(
            {
                "ok": "usable frontier, counted as positive",
                "judge_failed": "judge returned nothing usable",
                "not_degenerating": "judge says it is not a loop",
                "not_found": "quote could not be located in the tokens",
            }
        )

        by_split = (
            labels.groupby("split")
            .agg(
                rollouts=("is_positive", "size"),
                positives=("is_positive", "sum"),
                prompts=("prompt_id", "nunique"),
            )
            .reset_index()
        )
        by_split["positive rate"] = (
            by_split["positives"] / by_split["rollouts"]
        ).map("{:.2%}".format)

        onsets = labels.loc[labels["is_positive"], "onset_position"]
        shape = pd.DataFrame(
            [
                {
                    "": "where the loop starts, in tokens",
                    "min": int(onsets.min()),
                    "25%": int(onsets.quantile(0.25)),
                    "median": int(onsets.median()),
                    "75%": int(onsets.quantile(0.75)),
                    "max": int(onsets.max()),
                }
            ]
        )
        return by_split, resolution, shape

    splits_table, resolution_table, onset_shape = corpus_tables()

    mo.vstack(
        [
            mo.md(
                """
                Answers were generated from `Apertus-8B-Instruct-2509` at
                temperature 0.7, top-p 0.9, ten answers per prompt, with a
                4096-token limit. Prompts come from seven sources. Five are
                **in-domain** and are split by prompt into train, validation and
                an in-domain test set. Two, `codeforces` and `medical_o1`, are
                **held out** entirely and exist to ask whether anything
                transfers.
                """
            ),
            mo.ui.table(splits_table, selection=None),
            mo.md(
                """
                Only answers that hit the token limit are sent to the judge, and
                only those it can place a frontier in become positives. The rest
                are dropped from every split rather than being counted as either
                class, so nothing is trained or measured against a label nobody
                could check.
                """
            ),
            mo.ui.table(resolution_table, selection=None),
            mo.md(
                """
                The shape of the positives is the single most important fact
                about this dataset, and most design decisions follow from it.
                """
            ),
            mo.ui.table(onset_shape, selection=None),
            mo.md(
                """
                Every positive answer runs to the full 4096 tokens, and the loop
                typically starts about a fifth of the way in. So roughly four
                fifths of a positive answer is already loop. Telling such an
                answer from a healthy one is easy, and any measure that lets the
                deep-in-the-loop tokens dominate will read as almost perfect for
                almost any scorer. That is why the results below are split by
                *where* in the answer a token sits.

                Two consequences worth carrying around. `medical_o1` produces a
                single degenerate answer in six thousand, so it can never
                support a rate of its own. And the dropped answers are
                concentrated in one domain, `if_sft_data_verified`, so
                per-domain populations are not proportional to the corpus.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## What a probe is, and how it is trained

    The probe normalizes the residual stream at one layer and applies a linear
    map to a single score, 12,289 parameters. Activations are cached to disk
    once, with the language model frozen, so training reads them back and never
    runs the model. That is what makes wide sweeps affordable.

    Every run in this notebook shares the settings below. An experiment changes
    one of them at a time and says so; everything else is this.

    | | |
    |---|---|
    | features | read from the activation cache, model frozen |
    | depths | a separate head at every layer from 1 to 31, trained together |
    | optimizer | AdamW, learning rate 1e-4, no weight decay, linear decay to zero |
    | batch | 8 windows, accumulated to about 4096 target tokens per step |
    | length | 2000 steps, the cap, which every run reached |
    | validation | every 50 steps |
    | checkpoints | every 50 steps, all of them kept |
    | seeds | 42, 43, 44 wherever a claim ranks one recipe above another |

    The heads share no parameters and their losses are added, so each head's
    gradient is exactly what it would have been trained alone. Gradients are
    clipped per head rather than across all of them, which is what keeps that
    true.

    **Which answers evaluation reads is being changed, so it is worth keeping
    the three cases apart.** While these runs trained, evaluation read a sample
    of 400 validation answers, and at a 1% false-alarm rate that leaves the
    threshold pinned by a single healthy answer, which is most of why the
    quantity it steered on was unusable. Everything reported below, and the rule
    that decides which checkpoint counts, instead reads the **whole validation
    split**, written down in
    `configs/dataset/validation_rollouts_<dataset>.csv`. What evaluation during
    *future* training will read is still open: the whole split costs about ten
    minutes a pass against two for the sample, which is affordable only if heads
    stop early enough, and the replay is what will say whether they do.

    **The training answers are not the corpus.** Negatives outnumber positives
    by more than a hundred to one, so the training split is cut to four negative
    answers per positive, stratified by domain. Batches are then composed rather
    than shuffled: each batch is a quarter positive windows and three quarters
    negative, with negatives drawn in proportion to each domain's share.
    Validation and test are never subsampled this way.

    One consequence to keep in mind when reading run tables: an epoch here is
    one pass over the *positive* windows, not over the data. A rule that
    produces few windows per answer goes round many times; a rule that tiles
    every token does not finish a single pass.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## How a probe is measured

    All reported numbers come from the black-box evaluator. It takes one score
    per token and knows nothing about what produced them.

    **The operating point is a budget, never a threshold.** Raw thresholds mean
    nothing across scorers whose scores live on different scales. Instead we fix
    the share of healthy answers allowed to raise a false alarm, at 1%, 5% and
    10%, and read off the threshold that spends exactly that. Thresholds are
    chosen on validation, frozen to a file, and reused unchanged on test. The
    reporting tool refuses to produce a test report without that frozen file.

    **An alarm needs persistence.** A single token above threshold is usually
    noise, so an alarm requires $m$ consecutive tokens above it. The first alarm
    is

    $$a_r(\tau, m) = \min\{t : p_r(t') \ge \tau \ \text{for all} \ t' \in [t, t+m)\}$$

    Everything on disk today uses $m = 1$.

    ### The four views

    | | question |
    |---|---|
    | **A. Detection** | Does it fire on degenerate answers and stay quiet on good ones? |
    | **B. Coverage** | What fraction of tokens does it flag, split by where they sit? |
    | **C. Lead time** | How early or late is the alarm, relative to the true frontier? |
    | **D. Persistence** | Once it fires, does it keep firing, or was that a blip? |

    No view is read alone. A scorer that fires on every token has perfect
    recall, perfect coverage and perfect persistence, and only its false-alarm
    rate gives it away.

    View C reads in the direction that matters: **negative is early**. View D
    reads in opposite directions for the two populations. On positives a long
    alarm is good. On negatives every alarm is wrong, and its length separates a
    jittery scorer, which a larger $m$ would fix, from a confidently wrong one,
    which no $m$ can.

    ### The two coverage numbers

    View B splits by where a token sits relative to the frontier, and the split
    is the point.

    - **In-pattern coverage.** Of the tokens at or after the frontier, the share
      flagged. This asks whether the probe sees the obvious. It is the easy
      half.
    - **Warning coverage.** Of the tokens in a short band immediately *before*
      the frontier, the share flagged. This asks whether the probe sees the
      approach, which is the whole claim of the project. Reported over the 128
      and 256 tokens before the loop.

    Warning coverage is the number to rank on. Lead time answers the same
    question but is a median over the answers that happened to fire, which
    rewards a probe for missing the hard ones: drop the late-firing answers and
    the median improves. Warning coverage is defined over every positive answer,
    so an answer that is never flagged lowers it instead of disappearing from
    it. An answer whose loop starts at token 0 has no run-up and contributes to
    neither the hits nor the total.

    Lead time stays in View C, because it is the number a reader wants, and it
    is honest as a *result*. It just should not be the thing anything is
    selected on.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Which checkpoint gets reported

    This is currently the weakest link in the programme and it is worth stating
    plainly.

    A run trains for 2000 steps and keeps a checkpoint every 50. During training
    it watches one number on the monitor: the share of degenerate answers caught
    while holding false alarms at 1%. The checkpoint with the best value of that
    number is the one carried forward.

    **That number is saturated.** Across the forty evaluations of a run it takes
    two to four distinct values, because the monitor holds 108 positive answers
    so the metric can only move in steps of one in 108, and because almost every
    degenerate answer is caught almost immediately. Ranking quality over answers
    sits at 0.999 from the first evaluation onward.

    Two things follow. The selected checkpoint is usually step 50 to 150, so
    most of each run is discarded. And which of the tied checkpoints wins is
    arbitrary, which means it cannot separate recipes either.

    The population made it worse. These runs evaluated on a sample of 400
    validation answers, every degenerate one plus 292 healthy ones. At a 1%
    false-alarm rate that allows two false alarms, so the threshold sat on the
    third-highest healthy answer: one order statistic out of 292. Resampling
    which healthy answers land in that set moves anything computed from the
    threshold by 70% or more, which is why the rule below reads the whole
    validation split instead.

    ### The rule replacing it

    Applied to each depth independently, since the heads share no parameters and
    plateau at very different steps.

    - A head's **objective** is its coverage of the tokens immediately before
      the frontier, at a width $W$. This is what the project is about, it is an
      average over tens of thousands of tokens so it moves smoothly, and it is
      defined over every degenerate answer, so an answer that is never flagged
      lowers it rather than dropping out of it.
    - A head becomes **selectable** once its coverage inside the loop reaches a
      floor $g$. Below that a head has not learned to see the obvious and can
      post a flattering number before the frontier by accident.
    - **Progress** is tracked on whichever quantity the head still has to earn:
      coverage inside the loop while it is below the floor, the objective once
      it is above. Watching the objective too early would stop a head that is
      still learning, since a head that has not found the loop sits near zero
      before it and looks flat.
    - A head **stops** when the tracked quantity has not improved by more than
      $\epsilon$ for $P$ consecutive evaluations. This applies whether or not it
      ever became selectable: a head that plateaus below the floor has stopped
      learning, and since a run ends only when its slowest depth stops, a head
      that could never stop would hold the whole run open.
    - The checkpoint kept for a head is its best objective among the steps where
      it was selectable. A head that stopped without ever becoming selectable is
      recorded as such and left out of depth comparisons rather than quietly
      contributing its best-looking step.

    Coverage falls as well as rises, so selectability latches: a head near the
    floor would otherwise cross back and forth and restart its patience on every
    crossing.

    $W$, $g$, $\epsilon$ and $P$ are not guessed. Every checkpoint of every run
    was kept, so the whole trajectory can be recomputed and each combination
    read off it.

    ### Reproducing it on runs that already finished

    The runs in this notebook trained to the step cap, so the rule is applied to
    them afterwards, on the same evaluation population that future runs will
    use, and it decides the same thing it would have decided live.

    This is affordable because a frozen model makes the cached activations the
    same bytes for every checkpoint, and reading them is what a pass costs. So
    the activations are read once and every checkpoint is applied to them
    together, turning one pass per checkpoint into one pass per run. Two
    identities make that exact rather than approximate: normalising is the same
    work for every checkpoint, and a trained scale followed by a linear map is
    another linear map, so a depth's whole set of checkpoints collapses into one
    matrix multiply.

    The evaluation population is written down rather than resampled at runtime,
    in `configs/dataset/validation_rollouts_<dataset>.csv`, so a run and a later
    replay of it cannot silently disagree about what they measured.
    """)
    return


@app.cell
def _(Path, all_runs, missing, mo, pd):
    from degeneration_probe.evaluation.head_selection import (
        StoppingRule,
        apply_rule_to_run,
        run_length,
        sweep,
    )

    STEP_CAP = 2000

    def replayed_runs():
        """Runs whose saved checkpoints have been put through the rule."""
        if all_runs.empty:
            return {}
        found = {}
        for row in all_runs.itertuples():
            path = Path(row.run_dir) / "checkpoint_replay.parquet"
            if path.is_file():
                found[Path(row.run_dir).parent.name] = path
        return found

    def stopping_outcomes(rule=StoppingRule()):
        """Where each depth of each replayed run would have stopped."""
        replays = replayed_runs()
        if not replays:
            return missing(
                "No run has had its checkpoints replayed yet. Run "
                "`scripts/replay_checkpoints.py --run-dir outputs/<run>/latest`, "
                "which reads the pinned evaluation population once and applies "
                "every saved checkpoint to it."
            )
        rows = []
        for name, path in replays.items():
            outcomes = apply_rule_to_run(pd.read_parquet(path), rule)
            rows.append(
                {
                    "run": name[:52],
                    "depths": len(outcomes),
                    "never selectable": int((~outcomes["became_eligible"]).sum()),
                    "earliest stop": outcomes["stopped_at"].min(),
                    "median stop": outcomes["stopped_at"].median(),
                    "run would end at": run_length(outcomes, STEP_CAP),
                    "best depth": outcomes.loc[
                        outcomes["selected_value"].idxmax(), "layer"
                    ]
                    if outcomes["selected_value"].notna().any()
                    else None,
                }
            )
        return mo.vstack(
            [
                mo.md(
                    f"At floor {rule.floor}, width {rule.band}, tolerance "
                    f"{rule.tolerance}, patience {rule.patience}. A run ends when "
                    "its **slowest** depth stops, so that column is what a future "
                    "run's walltime has to cover."
                ),
                mo.ui.table(pd.DataFrame(rows), selection=None),
            ]
        )

    def parameter_sweep():
        """What each setting of the rule would have decided."""
        replays = replayed_runs()
        if not replays:
            return missing("Nothing to sweep until at least one run has been replayed.")
        name, path = next(iter(replays.items()))
        table = sweep(
            pd.read_parquet(path),
            floors=(0.2, 0.3, 0.4, 0.5),
            bands=(128, 256),
            tolerances=(0.0, 0.001, 0.005),
            patiences=(2, 3, 4, 6),
            cap=STEP_CAP,
        )
        return mo.vstack(
            [
                mo.md(
                    f"Every combination, on `{name[:52]}`. The four numbers are "
                    "chosen from this rather than assumed: what matters is that "
                    "the depth chosen and the step reached are stable across a "
                    "range of them, not that any one setting is optimal."
                ),
                mo.ui.table(table, selection=None),
            ]
        )

    return parameter_sweep, stopping_outcomes


@app.cell
def _(stopping_outcomes):
    stopping_outcomes()
    return


@app.cell
def _(parameter_sweep):
    parameter_sweep()
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## The experiment register

    Every experiment is defined once, here, as data. The run lists, the
    configuration summaries and the commands in each section below are all
    generated from it, so what the notebook shows is what was run.
    """)
    return


@app.cell
def _():
    # Depth is an axis of every result rather than something that distinguishes
    # one run from another: a run carries a head at every depth, and which one to
    # read is decided when a result is read.
    PROBED_LAYERS = list(range(1, 32))
    DEFAULT_LAYER = 12
    # The depth the adapted stage trains its single probe at. Stage 1's depth
    # profile is what should set it; until that profile exists this is provisional
    # and the adapted runs have to be read as such.
    CHOSEN_DEPTH = DEFAULT_LAYER
    SEEDS = [42, 43, 44]

    BASE = {
        "training.features.regime": "cached",
        "training.lora.enabled": "false",
        # The same tokens per optimizer step and the same step cap for every
        # recipe, so the total gradient is held constant and only the choice of
        # tokens varies. Large enough that the widest window still fits a whole
        # micro-batch inside one step.
        "training.budget.tokens_per_step": 4096,
        "training.runtime.max_steps": 2000,
        "training.runtime.per_device_train_batch_size": 8,
        "training.validation.strategy": "steps",
        "training.validation.steps": 50,
        "training.checkpoint.strategy": "steps",
        "training.checkpoint.steps": 50,
        # A checkpoint holds one small head per depth, so a run's worth costs a
        # few hundred megabytes and keeping all of them is what allows the
        # selection rule to be reconsidered without retraining.
        "training.checkpoint.save_total_limit": "null",
        "training.validation.max_rollouts": 400,
        # Scoring happens afterwards, from a checkpoint, one depth at a time.
        "training.validation.final_splits": "[]",
        "training.runtime.seed": 42,
        # Every setting any experiment varies is named even where it equals the
        # configured default, so a recipe is described the same way wherever it
        # appears and identical recipes collapse to one run.
        "training.probe.layers": "[" + ",".join(str(n) for n in PROBED_LAYERS) + "]",
        "training.selection.strategy": "frontier_window",
        "training.selection.window_size": 128,
        "training.selection.anchor": "centered",
        "training.selection.positive_fraction": 0.25,
        "training.label.family": "frontier_hard",
        "training.label.horizon": 0,
        "training.loss.name": "bce",
        "training.loss.bce.use_pos_weight": "true",
    }

    LADDER_RUNGS = [
        "all_tokens",
        "rollout_balanced",
        "random_window",
        "frontier_window",
        "frontier_window_hard_negative",
    ]
    # Window size is positional only for the rules that place a window
    # deliberately. For the first two it sets how tokens are tiled or how many
    # each answer contributes, which is a different question.
    POSITIONAL_RUNGS = [
        "random_window",
        "frontier_window",
        "frontier_window_hard_negative",
    ]
    WINDOWS = [64, 128, 256, 512]
    return (
        BASE,
        CHOSEN_DEPTH,
        DEFAULT_LAYER,
        LADDER_RUNGS,
        POSITIONAL_RUNGS,
        PROBED_LAYERS,
        SEEDS,
        WINDOWS,
    )


@app.cell
def _(BASE, CHOSEN_DEPTH, LADDER_RUNGS, POSITIONAL_RUNGS, SEEDS, WINDOWS):
    def recipe(**changes):
        return {**BASE, **changes}

    def _grid_runs():
        runs = []
        for strategy in LADDER_RUNGS:
            sizes = WINDOWS if strategy in POSITIONAL_RUNGS else [128]
            for window in sizes:
                for seed in SEEDS:
                    runs.append(
                        {
                            "label": f"{strategy}, W={window} (seed {seed})",
                            "overrides": recipe(
                                **{
                                    "training.selection.strategy": strategy,
                                    "training.selection.window_size": window,
                                    "training.runtime.seed": seed,
                                }
                            ),
                        }
                    )
        return runs

    def _horizon_runs():
        settings = [
            ("all_tokens", "centered", 128, [0, 256, 1024]),
            ("frontier_window", "centered", 512, [0, 128, 256]),
            # A trailing window holds only tokens before the frontier, so a
            # horizon of zero would leave it with no positive token at all.
            ("frontier_window", "trailing", 512, [128, 256, 512]),
        ]
        runs = []
        for strategy, anchor, window, horizons in settings:
            for horizon in horizons:
                for seed in SEEDS:
                    runs.append(
                        {
                            "label": (
                                f"{strategy}/{anchor} W={window}, "
                                f"horizon {horizon} (seed {seed})"
                            ),
                            "overrides": recipe(
                                **{
                                    "training.selection.strategy": strategy,
                                    "training.selection.anchor": anchor,
                                    "training.selection.window_size": window,
                                    "training.label.horizon": horizon,
                                    "training.runtime.seed": seed,
                                }
                            ),
                        }
                    )
        return runs

    def _soft_runs():
        return [
            {
                "label": f"soft, {decay} over {length} (seed {seed})",
                "overrides": recipe(
                    **{
                        "training.selection.window_size": 512,
                        "training.label.family": "frontier_soft",
                        "training.label.decay": decay,
                        "training.label.decay_length": length,
                        # A soft target has no class to weight.
                        "training.loss.bce.use_pos_weight": "false",
                        "training.runtime.seed": seed,
                    }
                ),
            }
            for decay in ("exponential", "linear")
            for length in (128, 256)
            for seed in SEEDS
        ]

    def _regression_runs():
        return [
            {
                "label": f"regression on repetition score (seed {seed})",
                "overrides": recipe(
                    **{
                        "training.label.family": "token_signal",
                        "training.label.signal": "repetition_score",
                        "training.loss.name": "mse",
                        "training.runtime.seed": seed,
                    }
                ),
            }
            for seed in SEEDS
        ]

    def _balance_runs():
        return [
            {
                "label": (
                    f"pos_weight {'on' if weight else 'off'}, "
                    f"positives {fraction} (seed {seed})"
                ),
                "overrides": recipe(
                    **{
                        "training.loss.bce.use_pos_weight": str(weight).lower(),
                        "training.selection.positive_fraction": fraction,
                        "training.runtime.seed": seed,
                    }
                ),
            }
            for weight in (True, False)
            for fraction in (0.25, 0.5)
            for seed in SEEDS
        ]

    def _adapted_runs():
        runs = []
        for seed in SEEDS:
            common = {
                "training.probe.layers": "null",
                "training.probe.layer": CHOSEN_DEPTH,
                "training.runtime.seed": seed,
            }
            runs.append(
                {
                    "label": f"adapted, LoRA over the probed depth (seed {seed})",
                    "overrides": recipe(
                        **common,
                        **{
                            "training.features.regime": "adapted",
                            "training.lora.enabled": "true",
                            "training.lora.layers": "all",
                            # The head starts as noise, so adapters that moved at
                            # the head's rate would rewrite the representation
                            # before the probe knew what it was looking for.
                            "training.optimizer.lora_learning_rate": 1e-5,
                        },
                    ),
                    "sbatch": "--time=08:00:00",
                }
            )
            runs.append(
                {
                    "label": f"frozen control at the same depth (seed {seed})",
                    "overrides": recipe(**common),
                }
            )
        return runs

    grid_runs = _grid_runs()
    horizon_runs = _horizon_runs()
    soft_runs = _soft_runs()
    regression_runs = _regression_runs()
    balance_runs = _balance_runs()
    adapted_runs = _adapted_runs()
    return (
        adapted_runs,
        balance_runs,
        grid_runs,
        horizon_runs,
        regression_runs,
        soft_runs,
    )


@app.cell
def _(
    adapted_runs,
    balance_runs,
    grid_runs,
    horizon_runs,
    regression_runs,
    soft_runs,
):
    EXPERIMENTS = [
        {
            "id": "S0",
            "title": "How far do model-free signals get?",
            "runs": [],
            "shell": ["sbatch cluster/baselines.sbatch"],
            "axes": [],
        },
        {
            "id": "S1",
            "title": "Which tokens should the probe learn from?",
            "runs": grid_runs,
            "shell": [],
            "axes": ["selection", "window"],
        },
        {
            "id": "S2a",
            "title": "Does labelling the run-up buy lead time?",
            "runs": horizon_runs,
            "shell": [],
            "axes": ["selection", "anchor", "window", "horizon"],
        },
        {
            "id": "S2b",
            "title": "A target that fades in rather than switching on",
            "runs": soft_runs,
            "shell": [],
            "axes": ["decay", "decay_length"],
        },
        {
            "id": "S2c",
            "title": "Regression instead of detection",
            "runs": regression_runs,
            "shell": [],
            "axes": ["label", "signal", "loss"],
        },
        {
            "id": "S2d",
            "title": "Class balance and calibration",
            "runs": balance_runs,
            "shell": [],
            "axes": ["pos_weight", "positive_fraction"],
        },
        {
            "id": "S3",
            "title": "Do adapters earn their cost?",
            "runs": adapted_runs,
            "shell": [],
            "axes": ["features", "layer"],
        },
        {
            "id": "S4",
            "title": "The held-out test, once",
            "runs": [],
            "shell": [],
            "axes": [],
        },
    ]
    BY_ID = {entry["id"]: entry for entry in EXPERIMENTS}
    return BY_ID, EXPERIMENTS


@app.cell
def _(EXPERIMENTS, mo, shlex):
    # Several experiments ask different questions of the same configuration: the
    # ladder's frontier_window at W=512 is also the horizon stage's zero-horizon
    # arm. A run's identity comes from its settings, so those are one run, planned
    # once and tagged for each experiment it belongs to.
    def build_plan():
        plan = {}
        for entry in EXPERIMENTS:
            for run in entry["runs"]:
                key = tuple(sorted(run["overrides"].items()))
                slot = plan.setdefault(
                    key,
                    {
                        "overrides": run["overrides"],
                        "exps": [],
                        "labels": [],
                        "sbatch": run.get("sbatch", ""),
                    },
                )
                if entry["id"] not in slot["exps"]:
                    slot["exps"].append(entry["id"])
                slot["labels"].append(f"{entry['id']}: {run['label']}")
        return list(plan.values())

    PLAN = build_plan()

    def render(slot):
        tags = ",".join(f'"exp:{exp}"' for exp in slot["exps"])
        overrides = [
            f"{key}={value}" for key, value in sorted(slot["overrides"].items())
        ]
        overrides.append(f"training.wandb.tags=[{tags}]")
        rendered = " ".join(shlex.quote(item) for item in overrides)
        flags = f"{slot['sbatch']} " if slot.get("sbatch") else ""
        return f"sbatch {flags}cluster/train.sbatch {rendered}"

    def commands_for(exp_id):
        entry = next(e for e in EXPERIMENTS if e["id"] == exp_id)
        return list(entry["shell"]) + [
            render(slot) for slot in PLAN if exp_id in slot["exps"]
        ]

    def show_commands(exp_id):
        lines = commands_for(exp_id)
        if not lines:
            return mo.md("_No commands: this stage trains nothing._")
        body = "\n".join(lines)
        return mo.accordion(
            {
                f"The exact commands ({len(lines)})": mo.md(
                    "```bash\ncd /iopsstor/scratch/cscs/mdenegri/degeneration-probe\n"
                    f"{body}\n```"
                )
            }
        )

    return PLAN, render, show_commands


@app.cell
def _(mo):
    mo.md("""
    ### Configuration summaries

    Each stage below opens with the settings it varies and the values it takes.
    These are read from the register, not written by hand.
    """)
    return


@app.cell
def _(BY_ID, mo):
    # How an override path is named when a configuration is described. Only the
    # settings an experiment actually varies are shown, so a summary stays short
    # enough to read.
    OVERRIDE_NAMES = {
        "training.selection.strategy": "selection",
        "training.selection.window_size": "window",
        "training.selection.anchor": "anchor",
        "training.selection.positive_fraction": "positive_fraction",
        "training.label.family": "label",
        "training.label.horizon": "horizon",
        "training.label.decay": "decay",
        "training.label.decay_length": "decay_length",
        "training.label.signal": "signal",
        "training.loss.name": "loss",
        "training.loss.bce.use_pos_weight": "pos_weight",
        "training.features.regime": "features",
        "training.probe.layer": "layer",
        "training.optimizer.lora_learning_rate": "lora_lr",
        "training.lora.layers": "lora_layers",
        "training.runtime.seed": "seed",
    }

    def config_summary(exp_id):
        """The settings this experiment varies, and the values it gives them."""
        runs = BY_ID[exp_id]["runs"]
        if not runs:
            return mo.md("_This stage trains nothing._")
        values = {}
        for run in runs:
            for path, value in run["overrides"].items():
                values.setdefault(path, []).append(value)
        lines = []
        for path, seen in values.items():
            distinct = sorted({str(v) for v in seen})
            if len(distinct) < 2 and path != "training.runtime.seed":
                continue
            name = OVERRIDE_NAMES.get(path, path)
            lines.append(f"| `{name}` | {', '.join(distinct)} |")
        if not lines:
            lines.append("| | nothing varies but the seed |")
        body = "\n".join(lines)
        return mo.md(
            "| setting | values |\n|---|---|\n"
            + body
            + "\n\nEverything else is the shared setup above."
        )

    return (config_summary,)


@app.cell
def _(OUTPUTS, json, mo, pd):
    def _checkpoint_step(path):
        if not path:
            return None
        tail = str(path).rsplit("checkpoint-", 1)
        return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None

    def load_all_runs():
        """One row per run attempt, read from what each run wrote about itself."""
        rows = []
        if not OUTPUTS.is_dir():
            return pd.DataFrame()
        for info_path in sorted(OUTPUTS.glob("*/*/run_info.json")):
            if info_path.parent.is_symlink():
                continue
            try:
                info = json.loads(info_path.read_text())
            except ValueError:
                continue
            training = info.get("training", {})
            axes = info.get("axes") or {}
            tags = info.get("tags") or []
            # A couple of settings an experiment varies are not in the recorded
            # axes, so they are read from the resolved configuration instead.
            selection = label = {}
            resolved = info_path.parent / "resolved_config.json"
            if resolved.is_file():
                try:
                    config = json.loads(resolved.read_text())["training"]
                    selection = config.get("selection", {})
                    label = config.get("label", {})
                except (ValueError, KeyError):
                    selection = label = {}
            rows.append(
                {
                    "exps": [
                        t.split(":", 1)[1] for t in tags if t.startswith("exp:")
                    ],
                    "group": info.get("group"),
                    "status": info.get("status"),
                    "steps": training.get("global_step"),
                    "epochs": training.get("epochs_completed"),
                    "selected_step": _checkpoint_step(training.get("best_checkpoint")),
                    "tokens_per_step": (training.get("budget") or {}).get(
                        "tokens_per_step_realized"
                    ),
                    "seed": axes.get("seed"),
                    "layer": axes.get("layer"),
                    "selection": axes.get("selection"),
                    "window": axes.get("window"),
                    "anchor": axes.get("anchor"),
                    "label": axes.get("label"),
                    "horizon": axes.get("horizon"),
                    # The recorded axis bakes the length into the shape, as
                    # `exponential128`. The length is read separately below, so
                    # only the shape is kept here.
                    "decay": (axes.get("decay") or "").rstrip("0123456789") or None,
                    "signal": axes.get("signal"),
                    "loss": axes.get("loss"),
                    "pos_weight": axes.get("pos_weight"),
                    "features": axes.get("features"),
                    "positive_fraction": selection.get("positive_fraction"),
                    "decay_length": label.get("decay_length"),
                    "run_dir": str(info_path.parent),
                }
            )
        return pd.DataFrame(rows)

    def missing(message):
        return mo.md(f"**Nothing to show yet.** {message}").callout(kind="warn")

    all_runs = load_all_runs()
    return all_runs, missing


@app.cell
def _(BY_ID, all_runs, missing, mo, pd):
    def tagged_with(exp_id):
        if all_runs.empty or "exps" not in all_runs.columns:
            return all_runs
        return all_runs[all_runs["exps"].apply(lambda tags: exp_id in tags)]

    def finished_rows(exp_id):
        found = tagged_with(exp_id)
        if found.empty or "status" not in found.columns:
            return pd.DataFrame()
        return found[found["status"] == "finished"]

    def config_of(row, axes):
        """A short readable name for one configuration, from the axes it varies."""
        parts = []
        for axis in axes:
            value = getattr(row, axis, None)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            parts.append(f"{value}" if axis == "selection" else f"{axis}={value}")
        return ", ".join(parts) if parts else "single configuration"

    def run_table(exp_id):
        """Which runs this experiment is, and how far each of them got."""
        axes = BY_ID[exp_id]["axes"]
        expected = len(BY_ID[exp_id]["runs"])
        found = tagged_with(exp_id)
        if found.empty:
            return missing(
                f"No run is tagged `exp:{exp_id}` yet. The register expects "
                f"{expected}."
            )
        rows = []
        for row in found.itertuples():
            rows.append(
                {
                    "configuration": config_of(row, axes),
                    "seed": row.seed,
                    "status": row.status,
                    "steps": row.steps,
                    "passes over the positives": (
                        round(row.epochs, 1) if row.epochs is not None else None
                    ),
                    "tokens per step": row.tokens_per_step,
                    "reported checkpoint": row.selected_step,
                }
            )
        frame = pd.DataFrame(rows).sort_values(["configuration", "seed"])
        done = int((found["status"] == "finished").sum())
        header = mo.md(
            f"**{done} of {expected} runs finished.** One row per run; the "
            "configurations are the distinct settings, each trained at three "
            "seeds."
        )
        return mo.vstack([header, mo.ui.table(frame, selection=None)])

    return config_of, finished_rows, run_table


@app.cell
def _(BASELINE_ROOT, Path, finished_rows, missing, pd):
    # A run holds a probe at every depth, so a depth is a directory inside the run
    # rather than a run of its own. Both readers below present a depth as a column,
    # so a single-depth run and a many-depth run can go on one axis.
    def own_depth(value):
        """The single depth a run trained, or None if it carries every depth."""
        return None if value is None or pd.isna(value) else int(value)

    def evaluation_dir(run_dir, own_layer, split, layer):
        root = Path(run_dir)
        own = own_depth(own_layer)
        if layer is not None:
            scoped = root / "layers" / f"layer_{int(layer):02d}" / "evaluation" / split
            if scoped.is_dir():
                return scoped
            # A run that trained a single depth keeps its output at the root, but
            # it can only answer for the depth it actually trained.
            if own is not None and own != int(layer):
                return None
        plain = root / "evaluation" / split
        return plain if plain.is_dir() else None

    def load_view(exp_id, split, view, layer):
        """One view of every scored run of an experiment, at one depth."""
        frames = []
        rows = finished_rows(exp_id)
        if rows.empty:
            return pd.DataFrame()
        for row in rows.itertuples():
            directory = evaluation_dir(row.run_dir, row.layer, split, layer)
            if directory is None or not (directory / f"{view}.csv").is_file():
                continue
            frame = pd.read_csv(directory / f"{view}.csv")
            frame["run_dir"] = row.run_dir
            frame["seed"] = row.seed
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load_baseline_view(split, view):
        """The same view for the model-free scorers, which have no run directory."""
        frames = []
        if not BASELINE_ROOT.is_dir():
            return pd.DataFrame()
        for directory in sorted(BASELINE_ROOT.glob("*/evaluation/" + split)):
            path = directory / f"{view}.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame["configuration"] = directory.parent.parent.name
            frame["seed"] = None
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def scored_depths(exp_id, split):
        depths = set()
        for row in finished_rows(exp_id).itertuples():
            root = Path(row.run_dir)
            for scoped in sorted(root.glob("layers/layer_*/evaluation")):
                if (scoped / split).is_dir():
                    depths.add(int(scoped.parent.name.split("_")[-1]))
            own = own_depth(row.layer)
            if own is not None and (root / "evaluation" / split).is_dir():
                depths.add(own)
        return sorted(depths)

    def nothing_scored(exp_id, split, layer):
        available = scored_depths(exp_id, split)
        where = (
            f" Depths scored so far: {available}."
            if available
            else " No depth of this experiment has been scored."
        )
        return missing(
            f"No finished run of {exp_id} has been through the evaluator on "
            f"`{split}` at layer {layer}.{where} Training writes no scores by "
            "itself; scoring is a separate job."
        )

    return load_baseline_view, load_view, nothing_scored, scored_depths


@app.cell
def _(BY_ID, config_of, finished_rows, load_baseline_view, load_view, pd):
    # The four views, as the columns worth putting side by side. Each entry is the
    # view it comes from, the column inside it, and how it reads.
    SCORECARD = [
        ("A", "recall", "view_a_detection", "recall", "share of degenerate answers caught"),
        ("A", "precision", "view_a_detection", "precision", "share of alarms that were real"),
        ("B", "in-pattern coverage", "view_b_coverage", "in_pattern_recall", "tokens flagged inside the loop"),
        ("B", "warning coverage 128", "view_b_coverage", "warning_recall_128", "tokens flagged in the 128 before it"),
        ("B", "warning coverage 256", "view_b_coverage", "warning_recall_256", "tokens flagged in the 256 before it"),
        ("B", "healthy token rate", "view_b_coverage", "token_false_positive_rate", "healthy tokens flagged"),
        ("C", "median lead", "view_c_lead_time", "median_offset", "tokens from alarm to loop, negative is early"),
        ("C", "never fired", "view_c_lead_time", "never_fired_positives", "degenerate answers never flagged"),
    ]
    # View D reports the two populations as separate rows, so it is picked out by
    # population rather than by column alone.
    PERSISTENCE = [
        ("D", "alarm length, degenerate", "positive", "tokens the first alarm holds"),
        ("D", "alarm length, healthy", "negative", "tokens a false alarm holds"),
    ]

    def _long_from(frame, key, name, column, budget):
        if frame.empty or column not in frame.columns:
            return []
        rows = frame[frame["target_negative_fpr"] == budget]
        return [
            {"configuration": r[key], "view": name[0], "metric": name[1], "value": r[column]}
            for _, r in rows.iterrows()
        ]

    def _probe_records(exp_id, split, layer, budget):
        rows = finished_rows(exp_id)
        if rows.empty:
            return []
        axes = BY_ID[exp_id]["axes"]
        naming = {r.run_dir: config_of(r, axes) for r in rows.itertuples()}
        records = []
        for view, metric, view_file, column, _ in SCORECARD:
            frame = load_view(exp_id, split, view_file, layer)
            if frame.empty:
                continue
            frame = frame.assign(
                configuration=frame["run_dir"].map(naming).fillna("unknown")
            )
            records += _long_from(frame, "configuration", (view, metric), column, budget)
        persistence = load_view(exp_id, split, "view_d_persistence", layer)
        for view, metric, population, _ in PERSISTENCE:
            if persistence.empty:
                continue
            frame = persistence[persistence["population"] == population]
            frame = frame.assign(
                configuration=frame["run_dir"].map(naming).fillna("unknown")
            )
            records += _long_from(
                frame, "configuration", (view, metric), "median_first_run_length", budget
            )
        return records

    def _baseline_records(split, budget):
        records = []
        for view, metric, view_file, column, _ in SCORECARD:
            base = load_baseline_view(split, view_file)
            if not base.empty:
                records += _long_from(base, "configuration", (view, metric), column, budget)
        persistence = load_baseline_view(split, "view_d_persistence")
        for view, metric, population, _ in PERSISTENCE:
            if persistence.empty:
                continue
            frame = persistence[persistence["population"] == population]
            records += _long_from(
                frame, "configuration", (view, metric), "median_first_run_length", budget
            )
        return records

    def scorecard_long(exp_id, split, layer, budget):
        """Every view of every configuration, one row per measurement.

        The model-free scorers ride along as reference rows, so a probe is never
        read except beside them. They are only added once the experiment has
        something of its own to compare, since a table holding nothing but the
        baselines would read as a result for a stage that has none. S0 is the
        exception, because there the baselines are the result.
        """
        records = _probe_records(exp_id, split, layer, budget)
        if records or exp_id == "S0":
            records = records + _baseline_records(split, budget)
        return pd.DataFrame(records)

    def is_baseline(name):
        return name in {"repetition", "entropy", "lrs"}

    # Columns read in the order the views are defined in, not in whatever order
    # the files happened to be loaded.
    METRIC_ORDER = [metric for _, metric, *_ in SCORECARD] + [
        metric for _, metric, *_ in PERSISTENCE
    ]

    return METRIC_ORDER, is_baseline, scorecard_long


@app.cell
def _(METRIC_ORDER, is_baseline, mo, nothing_scored, scorecard_long):
    def scorecard(exp_id, split, layer, budget):
        """The four views of every configuration, seeds collapsed to a spread."""
        long = scorecard_long(exp_id, split, layer, budget)
        if long.empty:
            return nothing_scored(exp_id, split, layer)

        def summarise(values):
            values = values.dropna()
            if values.empty:
                return ""
            middle = values.median()
            text = f"{middle:.3f}" if abs(middle) < 100 else f"{middle:.0f}"
            if len(values) > 1 and values.min() != values.max():
                low, high = values.min(), values.max()
                span = (
                    f"{low:.3f}–{high:.3f}"
                    if abs(middle) < 100
                    else f"{low:.0f}–{high:.0f}"
                )
                return f"{text} [{span}]"
            return text

        table = (
            long.pivot_table(
                index="configuration",
                columns="metric",
                values="value",
                aggfunc=summarise,
            )
            .reindex(columns=[m for m in METRIC_ORDER if m in set(long["metric"])])
            .reset_index()
        )
        # Model-free scorers first, so every probe is read against them.
        table["_order"] = table["configuration"].map(
            lambda name: (0 if is_baseline(name) else 1, name)
        )
        table = table.sort_values("_order").drop(columns="_order")
        return mo.vstack(
            [
                mo.md(
                    f"At a **{budget:.0%} false-alarm budget** on `{split}`, layer "
                    f"{layer}. Median across seeds, with the range across seeds in "
                    "brackets. Model-free scorers are the first rows."
                ),
                mo.ui.table(table, selection=None),
            ]
        )

    return (scorecard,)


@app.cell
def _(mpl, plt):
    # The documented reference palette, in its documented order. These hues are
    # validated as a set for line and dot charts; a chart that needs every pair to
    # be distinguishable at once stops at the first three.
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    # A sequential ramp for window size, which is an ordered quantity and should
    # not be given categorical hues.
    RAMP = ["#bdd7f2", "#7fb2e5", "#2a78d6", "#17457d"]
    INK = "#0b0b0b"
    INK_SOFT = "#52514e"
    INK_MUTED = "#8a8880"
    GRID = "#e3e2dd"

    def use_paper_style():
        mpl.rcParams.update(
            {
                "figure.dpi": 110,
                "savefig.bbox": "tight",
                "font.size": 9,
                "axes.titlesize": 9.5,
                "axes.labelsize": 9,
                "axes.titlelocation": "left",
                "axes.titlepad": 8,
                "axes.edgecolor": INK_MUTED,
                "axes.labelcolor": INK_SOFT,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "text.color": INK,
                "xtick.color": INK_SOFT,
                "ytick.color": INK_SOFT,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "legend.frameon": False,
                "grid.color": GRID,
                "grid.linewidth": 0.7,
                "lines.linewidth": 2.0,
                "lines.markersize": 5,
            }
        )

    use_paper_style()

    def save(fig, name, figures_dir):
        for suffix in (".pdf", ".png"):
            fig.savefig((figures_dir / name).with_suffix(suffix))
        return fig

    def tidy(axis, *, xgrid=True):
        axis.grid(axis="x" if xgrid else "y", alpha=0.9)
        axis.set_axisbelow(True)
        return axis

    def _unused_plt_guard():
        return plt

    return INK_MUTED, INK_SOFT, RAMP, SERIES, save, tidy


@app.cell
def _(
    FIGURES,
    INK_MUTED,
    INK_SOFT,
    SERIES,
    is_baseline,
    nothing_scored,
    plt,
    save,
    scorecard_long,
    tidy,
):
    # The four panels a stage is read on. Each is a dot per configuration with a
    # whisker for the seed spread, so a stage with a dozen configurations stays
    # readable where a dozen lines would not.
    PANELS = [
        ("A. does it fire on the right answers", ["recall", "precision"], None),
        (
            "B. what share of tokens does it flag",
            ["in-pattern coverage", "warning coverage 256"],
            None,
        ),
        ("C. how early, in tokens", ["median lead"], 0.0),
        ("D. how long the first alarm holds", ["alarm length, degenerate"], None),
    ]

    def views_figure(exp_id, split, layer, budget):
        long = scorecard_long(exp_id, split, layer, budget)
        if long.empty:
            return nothing_scored(exp_id, split, layer)

        names = sorted(
            long["configuration"].unique(),
            key=lambda n: (1 if is_baseline(n) else 0, n),
        )
        positions = {name: index for index, name in enumerate(names)}
        only_baselines = all(is_baseline(name) for name in names)

        fig, axes = plt.subplots(
            1, len(PANELS), figsize=(3.5 * len(PANELS), 0.34 * len(names) + 2.6)
        )
        for axis, (title, metrics, reference) in zip(axes, PANELS):
            drawn = False
            for colour, metric in zip(SERIES, metrics):
                subset = long[long["metric"] == metric]
                if subset.empty:
                    continue
                drawn = True
                grouped = subset.groupby("configuration")["value"]
                for name, values in grouped:
                    y = positions[name]
                    if values.min() != values.max():
                        axis.plot(
                            [values.min(), values.max()],
                            [y, y],
                            color=colour,
                            alpha=0.35,
                            linewidth=3,
                            solid_capstyle="round",
                        )
                    axis.plot(
                        values.median(),
                        y,
                        "o",
                        color=colour,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        label=metric if name == names[0] else None,
                    )
            if reference is not None:
                axis.axvline(reference, color=INK_MUTED, linewidth=1, zorder=0)
            axis.set_yticks(range(len(names)))
            axis.set_yticklabels(
                [n if not is_baseline(n) else f"{n} (no model)" for n in names]
            )
            axis.set_ylim(-0.7, len(names) - 0.3)
            axis.invert_yaxis()
            axis.set_title(title, color=INK_SOFT)
            tidy(axis)
            if drawn and len(metrics) > 1:
                # Under the panel rather than inside it: the marks sit at
                # whatever value the data takes, so any in-axes corner is one
                # result away from being covered up.
                axis.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.06),
                    ncol=len(metrics),
                    handletextpad=0.4,
                    columnspacing=1.2,
                )
        for axis in axes[1:]:
            axis.set_yticklabels([])
        # A depth only means something once a probe is in the picture. The
        # model-free scorers read no layer at all.
        where = f"{split}" if only_baselines else f"{split}, layer {layer}"
        fig.suptitle(
            f"{exp_id} on {where}, at a {budget:.0%} false-alarm budget",
            x=0.005,
            ha="left",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        return save(fig, f"{exp_id}_{split}_L{int(layer):02d}_views", FIGURES)

    return (views_figure,)


@app.cell
def _(FIGURES, Path, RAMP, finished_rows, missing, pd, plt, save, tidy):
    def curve_figure(exp_id, layer):
        """Is the run healthy, and did it have enough budget to settle?"""
        rows = finished_rows(exp_id)
        if rows.empty:
            return missing(f"No finished run for {exp_id} yet.")
        frames = []
        for row in rows.itertuples():
            path = Path(row.run_dir) / "history.parquet"
            if not path.is_file():
                continue
            history = pd.read_parquet(path)
            wanted = {
                f"val/layer{int(layer):02d}/loss_unweighted": "loss",
                f"val/layer{int(layer):02d}/prediction_std": "spread",
            }
            present = {k: v for k, v in wanted.items() if k in history.columns}
            if not present:
                continue
            piece = history[["step"] + list(present)].rename(columns=present)
            piece["seed"] = row.seed
            frames.append(piece.dropna())
        if not frames:
            return missing(
                f"No finished run for {exp_id} logged layer {layer}."
            )
        curves = pd.concat(frames, ignore_index=True)

        panels = [
            ("loss", "validation loss, class weight removed"),
            ("spread", "spread of the scores"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
        for axis, (column, title) in zip(axes, panels):
            band = curves.groupby("step")[column].agg(["min", "median", "max"])
            axis.fill_between(
                band.index, band["min"], band["max"], color=RAMP[0], alpha=0.6, linewidth=0
            )
            axis.plot(band.index, band["median"], color=RAMP[3])
            if column == "spread":
                axis.axhline(0.01, color="#e34948", linestyle="--", linewidth=1)
            axis.set_title(title)
            axis.set_xlabel("step")
            tidy(axis, xgrid=False)
        fig.suptitle(
            f"{exp_id}, layer {layer}: every run, median and range",
            x=0.005,
            ha="left",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return save(fig, f"{exp_id}_L{int(layer):02d}_curves", FIGURES)

    return (curve_figure,)


@app.cell
def _(DEFAULT_LAYER, PROBED_LAYERS, mo):
    split_choice = mo.ui.dropdown(
        options=["val", "test_indomain", "test_heldout_domains"],
        value="val",
        label="split",
    )
    layer_choice = mo.ui.dropdown(
        options={str(n): n for n in PROBED_LAYERS},
        value=str(DEFAULT_LAYER),
        label="depth",
    )
    budget_choice = mo.ui.dropdown(
        options={"1%": 0.01, "5%": 0.05, "10%": 0.10},
        value="1%",
        label="false-alarm budget",
    )
    mo.vstack(
        [
            mo.md("These three choices apply to every stage below."),
            mo.hstack(
                [split_choice, layer_choice, budget_choice], justify="start", gap=2
            ),
        ]
    )
    return budget_choice, layer_choice, split_choice


@app.cell
def _(BY_ID, PLAN, all_runs, mo, pd):
    def programme_table():
        rows = []
        for entry in BY_ID.values():
            planned = len({tuple(sorted(r["overrides"].items())) for r in entry["runs"]})
            tagged = (
                all_runs[all_runs["exps"].apply(lambda t: entry["id"] in t)]
                if not all_runs.empty
                else pd.DataFrame()
            )
            done = (
                int((tagged["status"] == "finished").sum()) if not tagged.empty else 0
            )
            rows.append(
                {
                    "stage": entry["id"],
                    "question": entry["title"],
                    "runs planned": planned or "none",
                    "runs finished": done,
                }
            )
        return pd.DataFrame(rows)

    mo.vstack(
        [
            mo.md(
                f"""
                ---
                # The programme

                Five stages, run in order, each ending in a decision the next one
                needs. Within a stage the runs are independent and can all be
                queued at once. Configurations shared between stages are trained
                once and tagged for each, so the {len(PLAN)} distinct
                configurations below cover every stage.
                """
            ),
            mo.ui.table(programme_table(), selection=None),
            mo.md(
                """
                Test splits are read only in S4, and no threshold and no choice
                of recipe is ever made against them.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S0. How far do model-free signals get?

    Before asking what a probe can do, it is worth knowing what can be done with
    no model at all. Three scorers go through the identical evaluator, see the
    same answers, get their thresholds frozen the same way, and are reported in
    the same four views.

    - **repetition**, a repetition score over a sliding window.
    - **lrs**, the longest repeated substring, as a step at the position the
      repeat begins.
    - **entropy**, the model's own predictive entropy, inverted, on the grounds
      that a loop is confident rather than uncertain.

    This is the bar every later result has to clear.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("S0")
    return


@app.cell
def _(budget_choice, layer_choice, scorecard, split_choice):
    scorecard("S0", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(budget_choice, layer_choice, split_choice, views_figure):
    views_figure("S0", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(
    BASELINE_ROOT,
    FIGURES,
    INK_MUTED,
    INK_SOFT,
    RAMP,
    SERIES,
    load_baseline_view,
    missing,
    np,
    pd,
    plt,
    save,
    tidy,
):
    def baseline_story():
        """What a repetition counter alone achieves, and where it runs out."""
        coverage = load_baseline_view("val", "view_b_coverage")
        detection = load_baseline_view("val", "view_a_detection")
        table_path = BASELINE_ROOT / "repetition" / "evaluation" / "val" / "rollout_table.csv"
        if coverage.empty or detection.empty or not table_path.is_file():
            return missing("The baselines have not been through the evaluator yet.")

        rollouts = pd.read_csv(table_path)
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))

        left = axes[0]
        for colour, budget in zip(RAMP[1:], sorted(rollouts["target_negative_fpr"].unique())):
            fired = rollouts[
                (rollouts["target_negative_fpr"] == budget)
                & rollouts["is_positive"]
                & rollouts["fired"]
            ]["offset"].to_numpy(dtype=float)
            if fired.size == 0:
                continue
            order = np.sort(fired)
            left.plot(
                order,
                np.arange(1, order.size + 1) / len(rollouts[
                    (rollouts["target_negative_fpr"] == budget) & rollouts["is_positive"]
                ]),
                color=colour,
                label=f"{budget:.0%} budget",
            )
        left.axvline(0, color=INK_MUTED, linewidth=1, zorder=0)
        left.set_xlim(-1200, 600)
        left.set_ylim(0, 1)
        left.set_xlabel("tokens from the alarm to the loop, negative is early")
        left.set_ylabel("share of degenerate answers alarmed")
        left.set_title("it fires early, and it misses some entirely", color=INK_SOFT)
        left.legend(loc="upper left")
        # Where each curve flattens is how many answers it caught at all. The gap
        # from there to the top is the share it never fired on, which is the half
        # of the picture a median lead time cannot show.
        left.annotate(
            "never fired",
            xy=(600, 0.83),
            xytext=(180, 0.90),
            color=INK_SOFT,
            fontsize=8,
            ha="left",
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8),
        )
        left.annotate(
            "the loop starts",
            xy=(0, 0.06),
            xytext=(60, 0.06),
            color=INK_SOFT,
            fontsize=8,
            ha="left",
        )
        tidy(left, xgrid=False)

        right = axes[1]
        merged = coverage[coverage["configuration"] == "repetition"].merge(
            detection[detection["configuration"] == "repetition"],
            on="target_negative_fpr",
            suffixes=("", "_a"),
        )
        series = [
            ("recall", "answers caught"),
            ("in_pattern_recall", "tokens inside the loop"),
            ("warning_recall_256", "tokens in the 256 before it"),
        ]
        for colour, (column, name) in zip(SERIES, series):
            if column not in merged.columns:
                continue
            right.plot(
                merged["target_negative_fpr"], merged[column], "o-", color=colour, label=name
            )
        right.set_xlabel("false-alarm budget on healthy answers")
        right.set_ylabel("share flagged")
        right.set_ylim(0, 1)
        right.set_title("the easy half and the hard half", color=INK_SOFT)
        right.legend(loc="upper left")
        tidy(right, xgrid=False)

        fig.suptitle(
            "S0: the repetition counter on validation", x=0.005, ha="left", fontsize=10
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return save(fig, "S0_val_repetition", FIGURES)

    baseline_story()
    return


@app.cell
def _(mo):
    mo.md("""
    **What this stage settles.**

    A repetition counter is a real detector, not a straw man. At a 1%
    false-alarm budget it catches just under half of all degenerate answers,
    with a ranking AUC of 0.90. It misses 56 of the 108 degenerate answers and
    covers 45% of the tokens inside the loop, so there is real room above it,
    but any probe has to clear this before it has shown anything.

    Its **positional** numbers, though, are not what they look like, and the
    next section is about why. Read as printed, the counter appears to fire a
    median of 66 tokens before the frontier. It does not. That figure comes
    from a score that is allowed to read the text that follows the token it is
    scoring, which nothing generating an answer live could do.

    **The other two baselines do not produce a usable operating point**, and the
    reason is in how they are defined rather than in the signals themselves.
    Because an answer's score is the highest score any of its tokens reaches,
    `lrs` puts 81% of healthy answers at the ceiling, since almost every answer
    contains some repeated substring, and `entropy` puts 98.6% there, since
    almost every answer contains one fully confident token. No threshold can
    spend a 1% budget against a tie that large, so both are reported at a
    threshold above their whole range, and both fire on nothing. They need a
    different aggregation before they can be compared with anything.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The counter is reading the text it is supposed to be predicting

    The repetition score at token $t$ is one minus the type-token ratio over
    $[t,\ t + 256)$: the window sits **in front of** the token it labels. So the
    score printed at a token two hundred positions before the loop was computed
    from a stretch of text that is mostly loop. It did not anticipate anything.
    It read the loop and wrote the answer back at an earlier position.

    Nothing generating an answer one token at a time could do this. At token $t$
    the tokens after $t$ do not exist. Every positional number the counter
    posts, its lead time and its coverage of the run-up, is therefore an upper
    bound that no deployment could reach.

    The size of that gap is worth knowing, so it is measured below rather than
    argued about. The honest version of the same statistic uses the window
    *behind* the token, $[t - 256,\ t)$, and that is exactly the value already
    stored 256 positions earlier, so it needs no recomputation: shifting the
    scores forward by the window size gives the score a live system could have
    had.

    **The score itself is left as it is.** It is what the labelling pipeline
    computes, several other things read it, and changing it would invalidate
    comparisons that already exist. What changes is how its positional numbers
    are read.

    Two things to carry forward. Detection is untouched, because an answer's
    score is the highest any of its tokens reaches and shifting does not move a
    maximum, so the counter's recall and precision are honest. And the
    comparison this stage exists for was unfair in the counter's favour: the
    probe reads the model's state at token $t$, which is built only from tokens
    up to $t$, so it never had this advantage. `lrs` has the same problem in a
    stronger form, since it searches the finished answer for its longest
    repeated substring and then marks where that repeat began.
    """)
    return


@app.cell
def _(BASELINE_ROOT, mo, np, pd):
    def lookahead_cost():
        """What the forward-looking window is worth, in the numbers it inflates."""
        from degeneration_probe.dataset_gen.label import DEFAULT_WINDOW_SIZE
        from degeneration_probe.evaluation.protocol import (
            coverage_window,
            threshold_for_budget,
        )

        path = BASELINE_ROOT / "repetition" / "scores" / "val.parquet"
        if not path.is_file():
            return mo.md("_The repetition baseline has not been scored yet._")
        frame = pd.read_parquet(path)
        window = DEFAULT_WINDOW_SIZE

        def behind(values):
            """The same statistic over the window behind the token."""
            values = np.asarray(values, dtype=np.float64)
            shifted = np.empty_like(values)
            shifted[window:] = values[:-window] if values.size > window else values[:0]
            shifted[: min(window, values.size)] = values[0]
            return shifted

        rows = []
        for name, read in (
            ("as computed, window ahead", lambda v: np.asarray(v, dtype=np.float64)),
            ("window behind the token", behind),
        ):
            positives = frame[frame["is_positive"]]
            negatives = frame[~frame["is_positive"]]
            tau, _ = threshold_for_budget(
                np.array([read(v).max() for v in negatives["scores"]]), 0.01
            )
            peaks = np.array([read(v).max() for v in positives["scores"]])
            inside_hits = inside_total = band_hits = band_total = 0
            offsets = []
            for values, onset in zip(positives["scores"], positives["onset_position"]):
                scored = read(values)
                onset = int(onset)
                inside = scored[coverage_window(scored.size, onset, None)]
                inside_hits += int((inside >= tau).sum())
                inside_total += inside.size
                band = scored[coverage_window(scored.size, onset, window)]
                band_hits += int((band >= tau).sum())
                band_total += band.size
                fired = np.flatnonzero(scored >= tau)
                if fired.size:
                    offsets.append(fired[0] - onset)
            offsets = np.asarray(offsets)
            rows.append(
                {
                    "repetition score": name,
                    "answers caught": int((peaks >= tau).sum()),
                    "coverage inside the loop": round(inside_hits / inside_total, 4),
                    f"coverage in the {window} before": round(band_hits / band_total, 4),
                    "median alarm, tokens from the loop": float(np.median(offsets)),
                    "alarms before the loop": f"{np.mean(offsets < 0):.1%}",
                }
            )
        return mo.vstack(
            [
                mo.ui.table(pd.DataFrame(rows), selection=None),
                mo.md(
                    f"""
                    Detection is identical, as it must be. Everything positional
                    collapses: coverage of the run-up falls by about four fifths,
                    and the median alarm moves from before the loop to well
                    inside it. The apparent head start was the window length.

                    Read against this, a probe does not have to beat 0.15
                    coverage of the run-up, it has to beat roughly 0.04, and it
                    does not have to fire 66 tokens early, it has to fire before
                    a counter that is already nearly two hundred tokens late.
                    """
                ),
            ]
        )

    lookahead_cost()
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S1. Which tokens should the probe learn from?

    An answer has thousands of tokens and the probe can only be shown a fraction
    of them. This stage asks which fraction, and how much context each piece
    should carry, together.

    The five rules form a ladder, each changing exactly one decision relative to
    the one before it:

    1. **all_tokens**, every token, tiled into windows. No choice at all.
    2. **rollout_balanced**, a fixed budget of tokens per answer, drawn at
       random from anywhere in it. Corrects the population, nothing else.
    3. **random_window**, the same budget, now contiguous. Adds context.
    4. **frontier_window**, the same window, placed on the frontier. Adds
       position.
    5. **frontier_window_hard_negative**, the same, but negative windows are
       aimed at spans that already look repetitive. Adds difficulty.

    Window size and rule are swept together rather than one after the other. An
    anchored window trades coverage for earliness and its size moves the probe
    along that same trade-off, so fixing a size first would assume the best size
    is the same for every rule. Size is swept only for the three rules that
    place a window deliberately; for the first two it controls how tokens are
    tiled or how many each answer contributes, which is a different question.

    **How it will be read.** Not on whether degenerate answers are caught, which
    every rule will do. On warning coverage and on the depth the signal lives
    at.
    """)
    return


@app.cell
def _(config_summary):
    config_summary("S1")
    return


@app.cell
def _(show_commands):
    show_commands("S1")
    return


@app.cell
def _(run_table):
    run_table("S1")
    return


@app.cell
def _(curve_figure, layer_choice):
    curve_figure("S1", layer_choice.value)
    return


@app.cell
def _(budget_choice, layer_choice, scorecard, split_choice):
    scorecard("S1", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(budget_choice, layer_choice, split_choice, views_figure):
    views_figure("S1", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(
    FIGURES,
    INK_SOFT,
    SERIES,
    budget_choice,
    finished_rows,
    load_view,
    missing,
    pd,
    plt,
    save,
    scored_depths,
    split_choice,
    tidy,
):
    def depth_figure(exp_id, split, budget):
        """Where in the network the signal lives, per selection rule."""
        depths = scored_depths(exp_id, split)
        rows = finished_rows(exp_id)
        if not depths or rows.empty:
            return missing(
                f"{exp_id} has no depth through the evaluator on `{split}` yet. "
                "This is the figure the stage exists to produce."
            )
        naming = {r.run_dir: (r.selection, r.window) for r in rows.itertuples()}
        records = []
        for depth in depths:
            frame = load_view(exp_id, split, "view_b_coverage", depth)
            if frame.empty:
                continue
            frame = frame[frame["target_negative_fpr"] == budget]
            for _, row in frame.iterrows():
                rule, window = naming.get(row["run_dir"], ("unknown", None))
                records.append(
                    {
                        "layer": depth,
                        "rule": rule,
                        "window": window,
                        "warning": row.get("warning_recall_256"),
                        "in_pattern": row.get("in_pattern_recall"),
                    }
                )
        if not records:
            return missing(f"No coverage recorded for {exp_id} on `{split}`.")
        frame = pd.DataFrame(records)
        windows = sorted(w for w in frame["window"].dropna().unique())
        fig, axes = plt.subplots(
            1, max(len(windows), 1), figsize=(3.3 * max(len(windows), 1), 3.2), squeeze=False
        )
        rules = sorted(frame["rule"].dropna().unique())
        for axis, window in zip(axes[0], windows):
            panel = frame[frame["window"] == window]
            for colour, rule in zip(SERIES, rules):
                line = panel[panel["rule"] == rule].groupby("layer")["warning"].median()
                if line.empty:
                    continue
                axis.plot(line.index, line.to_numpy(), color=colour, label=rule)
            axis.set_title(f"window {int(window)}", color=INK_SOFT)
            axis.set_xlabel("layer")
            tidy(axis, xgrid=False)
        axes[0][0].set_ylabel("warning coverage, 256 tokens")
        axes[0][-1].legend(loc="best")
        fig.suptitle(
            f"{exp_id}: where the approach is visible, {split}, {budget:.0%} budget",
            x=0.005,
            ha="left",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return save(fig, f"{exp_id}_{split}_depth", FIGURES)

    depth_figure("S1", split_choice.value, budget_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S2a. Does labelling the run-up buy lead time?

    The plain frontier label calls a token one position before the loop as
    innocent as one a thousand positions before. The horizon moves that boundary
    earlier, asking the probe to fire while the text still looks fine. Whether
    that actually produces an earlier alarm is the whole of its claim.

    It is asked of three settings that differ in how much run-up the probe ever
    sees, because that is what decides how much the horizon can do. Under
    `all_tokens` a horizon of 256 adds about seven percent more positive tokens,
    since the loop starts early and most of a degenerate answer is already loop.
    Under a centred window of 512 the same horizon turns a window that was half
    positive into one that is entirely positive. A trailing window sits wholly
    before the frontier, so the horizon controls its entire positive class.

    A window has to be wide enough to express the horizon, or the comparison
    measures the window instead. A centred window spends half its length past
    the frontier and so needs a width of at least twice the horizon; a trailing
    window needs at least the horizon. Below that, two different horizons label
    every token in the window positive and train on identical data.

    **How it will be read.** On warning coverage and on the *distance* from the
    frontier, not on signed lead alone. If the frontier is where degeneration
    begins, firing five hundred tokens early is a false alarm inside a
    degenerate answer rather than an achievement.
    """)
    return


@app.cell
def _(config_summary):
    config_summary("S2a")
    return


@app.cell
def _(show_commands):
    show_commands("S2a")
    return


@app.cell
def _(run_table):
    run_table("S2a")
    return


@app.cell
def _(budget_choice, layer_choice, scorecard, split_choice):
    scorecard("S2a", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(
    FIGURES,
    INK_MUTED,
    INK_SOFT,
    SERIES,
    budget_choice,
    finished_rows,
    layer_choice,
    load_view,
    missing,
    pd,
    plt,
    save,
    split_choice,
    tidy,
):
    def horizon_figure(split, layer, budget):
        """What the horizon actually moves, if anything."""
        rows = finished_rows("S2a")
        if rows.empty:
            return missing("No finished run for S2a yet.")
        naming = {
            r.run_dir: (f"{r.selection}/{r.anchor} W={r.window}", r.horizon)
            for r in rows.itertuples()
        }
        panels = [
            ("view_b_coverage", "warning_recall_256", "warning coverage, 256 tokens", None),
            ("view_c_lead_time", "median_offset", "median lead, tokens", 0.0),
            ("view_c_lead_time", "never_fired_positives", "degenerate answers missed", None),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
        drew = False
        for axis, (view, column, title, reference) in zip(axes, panels):
            frame = load_view("S2a", split, view, layer)
            if frame.empty or column not in frame.columns:
                axis.set_axis_off()
                continue
            drew = True
            frame = frame[frame["target_negative_fpr"] == budget]
            tidy_rows = [
                {
                    "setting": naming.get(r["run_dir"], ("unknown", None))[0],
                    "horizon": naming.get(r["run_dir"], ("unknown", None))[1],
                    "value": r[column],
                }
                for _, r in frame.iterrows()
            ]
            table = pd.DataFrame(tidy_rows)
            for colour, setting in zip(SERIES, sorted(table["setting"].unique())):
                line = (
                    table[table["setting"] == setting]
                    .groupby("horizon")["value"]
                    .median()
                    .sort_index()
                )
                axis.plot(line.index, line.to_numpy(), "o-", color=colour, label=setting)
            if reference is not None:
                axis.axhline(reference, color=INK_MUTED, linewidth=1, zorder=0)
            axis.set_xlabel("horizon, tokens labelled positive before the loop")
            axis.set_title(title, color=INK_SOFT)
            tidy(axis, xgrid=False)
        if not drew:
            plt.close(fig)
            return missing(
                f"S2a has nothing through the evaluator on `{split}` at layer {layer}."
            )
        axes[0].legend(loc="best")
        fig.suptitle(
            f"S2a on {split}, layer {layer}, {budget:.0%} budget",
            x=0.005,
            ha="left",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return save(fig, f"S2a_{split}_L{int(layer):02d}_horizon", FIGURES)

    horizon_figure(split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S2b. A target that fades in rather than switching on

    The same frontier, with the step replaced by a decay, so that closer to the
    loop means more degenerate without committing to a hard cut. The decay
    length says how far back the run-up is taken to reach.

    The same width constraint applies as for the horizon. A centred window shows
    half its width of run-up, so a decay longer than that is a constant over
    everything the probe sees. These run at a width of 512 with decay lengths of
    128 and 256 for that reason.

    The class weight is off here, because a soft target has no class to weight.
    That makes the training loss incomparable with the hard-label runs, so only
    the protocol views can be read across the two.

    **How it will be read.** On warning coverage. If anchored rules turn out to
    win by reading the approach better rather than by reading the loop
    differently, a target that grades the approach aims at the same thing from
    the label side, and is the most likely of the target axes to move something.
    """)
    return


@app.cell
def _(config_summary):
    config_summary("S2b")
    return


@app.cell
def _(show_commands):
    show_commands("S2b")
    return


@app.cell
def _(run_table):
    run_table("S2b")
    return


@app.cell
def _(budget_choice, layer_choice, scorecard, split_choice):
    scorecard("S2b", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(budget_choice, layer_choice, split_choice, views_figure):
    views_figure("S2b", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S2c. Regression instead of detection

    A target that exists everywhere, including on healthy answers, where a
    nonzero repetition score describes text that is genuinely repetitive and
    perfectly legitimate.

    This trains a different concept from "this answer has broken", which is
    exactly why it is worth asking rather than assuming. A probe that tracks
    repetition will fire on a numbered list; a probe that tracks degeneration
    should not.

    **How it will be read.** On the healthy token rate at a fixed budget, not on
    recall. The expectation is better coverage inside the loop, where the target
    is dense, and worse behaviour on legitimate repetition.
    """)
    return


@app.cell
def _(config_summary):
    config_summary("S2c")
    return


@app.cell
def _(show_commands):
    show_commands("S2c")
    return


@app.cell
def _(run_table):
    run_table("S2c")
    return


@app.cell
def _(budget_choice, layer_choice, scorecard, split_choice):
    scorecard("S2c", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S2d. Class balance and calibration

    Imbalance can be corrected in how the training stream is composed, or in the
    loss, and doing both at once corrects it twice. This crosses the class
    weight with the fraction of positive windows.

    The expectation is that nothing moves. Both knobs mostly shift where the
    scores sit rather than how well they separate, and a threshold re-derived
    per run absorbs a shift. It is worth measuring anyway because "nothing
    moves" is a prediction, it costs twelve runs, and leaving it unmeasured
    means every later result carries an untested assumption.

    **How it will be read.** On the spread of the scores as much as on the
    views. A probe whose scores collapse toward a constant converges nicely and
    distinguishes nothing, and that failure is invisible in the loss.
    """)
    return


@app.cell
def _(config_summary):
    config_summary("S2d")
    return


@app.cell
def _(show_commands):
    show_commands("S2d")
    return


@app.cell
def _(run_table):
    run_table("S2d")
    return


@app.cell
def _(curve_figure, layer_choice):
    curve_figure("S2d", layer_choice.value)
    return


@app.cell
def _(budget_choice, layer_choice, scorecard, split_choice):
    scorecard("S2d", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S3. Do adapters earn their cost?

    Everything so far reads a frozen representation. If the run-up carries a
    signal the probe cannot reach, the limit might be the representation rather
    than the label or the selection rule. Low-rank adapters let the
    representation move, at a cost per configuration orders of magnitude higher.

    **It runs at one depth.** Sweeping depth inside a run works because the
    cached activations hold every layer. An adapted run has no cached
    activations: it runs the model, and adapters placed below one depth rewrite
    what every depth above it reads. So the heads stop being independent, and an
    adapted run carries one probe at one depth. The depth should come from S1's
    depth profile, which does not exist yet, so these runs are provisional.

    **Its control is its own run.** The comparison needs a frozen run at the
    same single depth, trained the same way, rather than one head of a
    many-headed run. A many-headed run picks its checkpoint on whichever depth
    leads and fits one class weight across all of them, so it differs from an
    adapted run in more than the regime.

    **The adapters move slowly on purpose.** A randomly initialised probe pushes
    gradients into the adapters before it knows what it is looking for, which
    rewrites the representation while the probe is still noise. The adapters
    therefore learn an order of magnitude more slowly than the head. Freezing
    them outright for the first few hundred steps is the stronger version of the
    same fix, and is the next thing to try if this is not enough.
    """)
    return


@app.cell
def _(config_summary):
    config_summary("S3")
    return


@app.cell
def _(show_commands):
    show_commands("S3")
    return


@app.cell
def _(run_table):
    run_table("S3")
    return


@app.cell
def _(curve_figure, layer_choice):
    curve_figure("S3", layer_choice.value)
    return


@app.cell
def _(budget_choice, layer_choice, scorecard, split_choice):
    scorecard("S3", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(budget_choice, layer_choice, split_choice, views_figure):
    views_figure("S3", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S4. The held-out test, once

    Everything up to here is chosen on validation data. This is the only number
    that says how the winner behaves on data nothing was tuned against.

    Nothing is trained. The winning recipe is scored on the two test splits
    using the thresholds already frozen on validation, applied unchanged. The
    reporting tool refuses to produce a test report for a scorer with no frozen
    thresholds, so the leak is structurally impossible rather than merely
    discouraged.

    Held-out domains are reported per domain and never pooled. Beside the
    frozen-threshold numbers sits the threshold-free ranking for each domain,
    because the two together separate a calibration shift, where the ordering
    still works and the threshold no longer fits, from a representation failure,
    where the ordering itself does not transfer. Those call for different fixes
    and one number cannot tell them apart. Any per-domain cell backed by fewer
    than ten degenerate answers is marked as anecdotal rather than quoted as a
    rate, which `medical_o1`, with one, always will be.

    After this, nothing is tuned. A second pass over the test splits with a
    different recipe would make them a validation set with extra steps.
    """)
    return


@app.cell
def _(budget_choice, layer_choice, scorecard, split_choice):
    scorecard("S4", split_choice.value, layer_choice.value, budget_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Running any of this

    Jobs go through Slurm. The login node is shared, so nothing heavier than a
    file listing runs there directly.

    ```bash
    cd /iopsstor/scratch/cscs/mdenegri/degeneration-probe

    # One training run. Hydra overrides pass straight through.
    sbatch cluster/train.sbatch training.selection.window_size=256

    # Score some depths of a finished run, and put each through the evaluator.
    sbatch cluster/score_layers.sbatch outputs/<run>/latest "8 12 30" val

    # Score one checkpoint of one depth, kept beside the others.
    python scripts/score_rollouts.py --run-dir outputs/<run>/latest --checkpoint checkpoint-450 --layer 12 --splits val --output-dir outputs/<run>/latest/sweep/layer_12/step_450

    # Put every saved checkpoint of a run through the stopping rule's measurements.
    sbatch cluster/replay.sbatch outputs/<run>/latest
    ```

    **Where a run lands.** Every run derives its name from its settings: the
    axes a person scans for, then the seed, then a fingerprint of the whole
    configuration, so two runs differing in any setting can never collide.
    Re-running a configuration adds a timestamped attempt beside the first
    rather than overwriting it.

    ```
    outputs/<run_name>/<timestamp>/     one attempt
    outputs/<run_name>/latest           symlink to the newest attempt
      run_info.json                     identity, axes, tags, status, timing
      dataset_summary.json              composition of every split
      history.parquet, metrics.jsonl    every logged step
      checkpoint-<step>/                one small head per depth
      layers/layer_NN/                  one depth, once it has been scored
        scores/<split>.parquet          one score per token per answer
        evaluation/<split>/*.csv        the four views
      decision_thresholds.json          frozen on validation, reused on test
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Notes for whoever picks this up next

    Facts that are expensive to rediscover.

    **Never run training or scoring on the login node.** Everything heavy goes
    through `sbatch`. File listings, `git`, the unit tests and
    `evaluate_scores.py` are fine directly.

    **Two Python environments.** `.venv/bin/python` at the repo root runs the
    tests and the cheap analysis scripts. Cluster jobs run inside a container
    described by `cluster/env.toml`, built from `cluster/Dockerfile` via
    `cluster/build.sh`.

    **The test suite is the fastest way to check a change.**
    `.venv/bin/python -m pytest tests -q`, about 40 seconds.

    **Activations.** One file per answer, shaped `[33, tokens, 4096]` in fp16,
    5.3 TB in total. **Slot 0 is the embedding**, so probe layer $L$ is cached
    slot $L + 1$. Getting this wrong silently trains on the neighbouring layer,
    which works well enough to look plausible. Every file carries the ordering
    in its own metadata and it is checked on read.

    **Ground truth.** The frontier is the judge's onset quote, located in the
    token stream and cached in `onset_labels/onset_quote_positions.parquet`;
    `onset_labels.parquet` is what training reads. One function owns the
    definition of an onset and nothing else should read one out of any other
    column. The pinned evaluation population follows the same principle: it
    stores only the three fields that locate a rollout's activations, so it
    cannot drift out of step with the labels.

    **Where the checkpoint rule lives.**
    `degeneration_probe/evaluation/head_selection.py` holds both the record one
    depth produces at one step and the rule applied to a sequence of them, as
    pure functions. Evaluation during training and the replay of saved
    checkpoints both call them, which is what makes the two agree by
    construction rather than by care.

    **Known traps.**

    - A trailing window with horizon 0 contains no positive token. The dataset
      refuses to build it rather than training on nothing.
    - `val/loss` carries a class weight fitted to each recipe's own training
      stream, so it is not comparable between recipes. Use
      `val/loss_unweighted`.
    - The token budget per step is derived from the *measured* tokens per
      example, not from the configured window size. Wide windows are clipped by
      the ends of the answers they sit in, so they land furthest below the
      request. The run table reports what each step really saw.
    - Most runs were interrupted by the walltime and resumed, so a run's
      recorded duration covers only its final leg, not its real cost.
    - `run_info.json` does not record the positive fraction or the decay length,
      so those are read back out of `resolved_config.json`.

    **Where else to look.** `notebooks/inspect_runs.py` is the inventory of
    every run regardless of experiment. `notebooks/rollout_metric_explorer.py`
    plots a single answer's metrics against its frontier, which is the fastest
    way to sanity-check a label by eye.
    """)
    return


@app.cell
def _(EXPERIMENTS, PLAN, REPO, mo, render):
    def write_launcher():
        lines = ["#!/bin/bash", "set -euo pipefail", f"cd {REPO}", ""]
        for entry in EXPERIMENTS:
            if entry["shell"]:
                lines.append(f"# --- {entry['id']}: {entry['title']}")
                lines.extend(entry["shell"])
                lines.append("")
        lines.append(f"# --- {len(PLAN)} training runs")
        for slot in PLAN:
            lines.append(f"# {'; '.join(slot['labels'])}")
            lines.append(render(slot))
        lines.append("")
        path = REPO / "cluster" / "launch_experiments.sh"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    launcher_path = write_launcher()
    mo.md(
        f"""
        ---
        Every command in this notebook, written to one script:

        ```
        {launcher_path}
        ```

        It is regenerated whenever this cell runs, so the register above stays
        the single source of truth.
        """
    )
    return


if __name__ == "__main__":
    app.run()
