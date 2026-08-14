import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    import shlex
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    REPO = Path("/iopsstor/scratch/cscs/mdenegri/degeneration-probe")
    OUTPUTS = REPO / "outputs"
    BUILD_ROOT = Path(
        "/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe"
        "/degeneration-dataset-apertus-8b-instruct"
    )
    FIGURES = REPO / "notebooks" / "figures" / "diary"
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES, OUTPUTS, Path, REPO, json, mo, pd, plt, re, shlex


@app.cell
def _(mo):
    mo.md("""
    # Detecting degeneration before it happens

    A running record of every experiment on the degeneration probe: what
    each one asks, the exact command that produced it, and its results as
    they arrive.

    Read it top to bottom the first time. The opening sections describe the
    system and how to read its numbers; everything after that is one
    experiment per section, each self-contained and each reproducible from
    the command printed beside it.

    Sections whose runs have not finished show what is missing rather than
    failing. Nothing here recomputes anything: every table is read from a
    run directory, which is why those directories were made
    self-describing.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## What the system is

    Large language models sometimes fall into a loop: partway through
    generating an answer they begin repeating a phrase, and never stop
    until they hit the token limit. The question here is whether that
    failure is visible **inside the model's own activations, while it is
    happening**, early enough to be worth acting on.

    The system has six parts.

    **1. A corpus of rollouts.** Answers generated from an 8B model, each
    one either ending naturally or hitting the token cap while looping. A
    rollout that ended on its own is a **negative**; one that was cut off
    mid-loop is a **positive**.

    **2. A ground-truth frontier.** For every positive rollout there is one
    token position where the degeneration begins, called the **frontier**.
    It comes from an LLM judge, which reads the rollout and names the spot
    by quoting the text; a separate step locates that quote in the token
    stream and records the index of its first token. Asking for a quote
    rather than a number is what makes the label checkable, both
    automatically against the completion and by a person reading it.

    **3. Two ways to read the model.** The probe looks at the residual
    stream at one layer. Either *adapted*, meaning the language model runs
    with LoRA adapters so the representation itself can change, or
    *cached*, meaning activations already written to disk are read back and
    only a small head is trained. Cached is roughly ten times faster and is
    what makes wide sweeps affordable.

    **4. The probe.** A linear head on one layer, emitting a score in
    $[0, 1]$ for every token. About twelve thousand parameters.

    **5. A rule for which tokens to train on.** A rollout has thousands of
    tokens and almost all of them are uninformative. There is a ladder of
    five rules, from "use everything" up to "use a window around the
    frontier, and choose negative windows that already look repetitive".
    Batches are composed to a fixed class mix and an equal token budget per
    step, so the rules can be compared without compute becoming the
    difference between them.

    **6. An evaluation protocol that never sees the probe.** A run writes
    one score per token per rollout to a file, and the evaluator reads only
    that file. Probes and simple heuristic baselines therefore go through
    exactly the same judgement. It reports four views:

    | View | The question it answers |
    |---|---|
    | **A. Detection** | Does it fire on degenerate rollouts and stay quiet on good ones? |
    | **B. Coverage** | Once inside the loop, what fraction of tokens does it flag? |
    | **C. Lead time** | How early or late is the alarm, relative to the true frontier? |
    | **D. Persistence** | When it fires, does it keep firing, or was that a one-token blip? |

    Thresholds are chosen on validation only and frozen before any test
    split is touched. That boundary is enforced in code, not by discipline:
    the function that picks thresholds refuses to run on anything but
    validation, and the function that applies them refuses to run without a
    frozen file to read.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Note: everything that can be varied (skip on a first read)": mo.md(
                """
                The knobs below are grouped by what a change **costs**, which
                matters more in practice than what it changes.

                **Free. One scores file answers these in seconds, no retraining.**

                - *Threshold budget.* A threshold is never picked directly.
                  Instead a tolerated false-alarm rate on negative rollouts is
                  picked (1%, 5%), and the threshold that spends exactly that
                  is read off.
                - *Persistence $m$.* Require the score to stay above threshold
                  for $m$ consecutive tokens before calling it an alarm.
                  Suppresses one-token spikes, at the cost of $m$ tokens of
                  delay.
                - *Split and per-domain slicing.* Which population is reported,
                  with a minimum positive count so thin cells stay unreported.

                **Cheap. Retrains the head only, about seven minutes.**

                *Features*

                - **Layer** (default 30 of 32). Which depth is read.
                  Activations are cached for every layer, so this sweep costs
                  only training time.
                - **Context window size** (default 1). Whether the head sees
                  one token or a short strip of recent ones. Repetition is
                  inherently a multi-token pattern.
                - **Normalization** of activations before the linear head.

                *Targets*

                - **Label family**: `frontier_hard` (a step at the frontier),
                  `frontier_soft` (a ramp leading up to it), `token_signal`
                  (ignore the frontier, regress a continuous signal).
                - **Horizon $N$** (default 0). How many tokens *before* the
                  frontier still count as positive. This is the lead-time dial.
                - **Decay shape and length** for soft labels.
                - **Signal** (repetition score or entropy) for the regression
                  family.

                *Which tokens*

                - **Strategy**, the five rungs: `all_tokens` →
                  `rollout_balanced` → `random_window` → `frontier_window` →
                  `frontier_window_hard_negative`. Each rung changes exactly
                  one decision relative to the previous one.
                - **Window size $W$**. More context per example, fewer
                  independent examples for the same budget.
                - **Anchor**: whether the frontier window sits entirely before
                  the frontier (`trailing`) or straddles it (`centered`).
                - **Positive fraction**: the share of windows drawn from
                  positive rollouts.
                - **Hard-negative fraction**: on the top rung, how much of the
                  negative budget aims at already-repetitive spans.
                - **Negatives per positive**, upstream at the rollout-sampling
                  level.

                *Optimization*

                - **Loss**: cross-entropy for binary targets, squared error for
                  the regression family.
                - **Class weight** on the positive class, and whether it is
                  applied at all.
                - **Learning rates** (head and adapters separately), weight
                  decay.
                - **Tokens per step**, the equal-budget rule.
                - **Steps**, patience, and the collapse threshold.
                - **Seed**. Not a design choice, but it is what produces a
                  noise floor, so it belongs on the list.

                **Expensive. Retrains with the model loaded, about 76 minutes.**

                - **Adapted instead of cached**, which unlocks the adapter
                  knobs: which layers adapt, rank, alpha, dropout.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Running anything here

    Jobs go through Slurm. The login node is shared, so nothing heavier
    than a file listing runs there directly.

    ```bash
    cd /iopsstor/scratch/cscs/mdenegri/degeneration-probe

    # One training run. Hydra overrides pass straight through.
    sbatch cluster/train.sbatch training.probe.layer=20

    # Score every token of every rollout for a finished run.
    sbatch cluster/score.sbatch --run-dir outputs/<run_name>/<attempt>

    # Turn those scores into the four views (cheap, runs on the login node).
    .venv/bin/python scripts/evaluate_scores.py --run-dir outputs/<run_name>/<attempt>
    ```

    Scoring depends on training, so the two are chained rather than
    watched:

    ```bash
    JOB=$(sbatch --parse cluster/train.sbatch <overrides>)
    sbatch --dependency=afterok:$JOB cluster/score.sbatch --run-dir <run_dir>
    ```

    **Where a run lands.** Every run derives its own name from its
    settings: the axes a human scans for, then the seed, then a
    fingerprint of the entire configuration. Two runs differing in any
    setting can never collide. Re-running the same configuration adds a
    timestamped attempt beside the first rather than overwriting it.

    ```
    outputs/<run_name>/<timestamp>/     one attempt
    outputs/<run_name>/latest           symlink to the newest attempt
      run_info.json                     identity, axes, tags, status, timing
      dataset_summary.json              composition of every split, class weight
      history.parquet, metrics.jsonl    every logged step
      scores/<split>.parquet            one score per token per rollout
      evaluation/<split>/*.csv          the four views
      decision_thresholds.json          frozen on validation, reused on test
    ```

    The **group** is the run name with the seed removed, so repeats of one
    recipe aggregate into a single line with a spread. Each experiment
    below also tags its runs with `exp:<id>`, which is how this notebook
    finds them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading the numbers, part 1: during training

    These come from the monitoring split at every validation step. They
    answer "is this run healthy and improving", not "is this probe good".

    | Metric | What it means | How to read it |
    |---|---|---|
    | `val/recall_at_budget` | Of the rollouts that really degenerate, the share caught while holding false alarms at 1% | **The selection metric.** Ties a checkpoint to the point it would actually be used at |
    | `val/budget_tau` | The threshold that 1% of false alarms buys | Rises when a negative rollout is scored confidently and wrongly, which is what costs recall |
    | `val/rollout_auc` | Ranking quality over rollouts, using each rollout's peak score | Rank-based, so unaffected by class weighting. **Saturates:** it reaches its ceiling long before the probe is useful, so read it as a health check, not a comparison |
    | `val/rollout_ap` | Average precision over the same ranking | More sensitive than AUC when positives are rare |
    | `val/loss` | The training objective, evaluated on validation | Weighted by a class weight fitted to *this recipe's* training stream. **Not comparable between recipes** |
    | `val/loss_unweighted` | The same loss with the weight removed | Measured identically for every recipe, so this is the one to compare across runs |
    | `val/prediction_std` | Spread of the probe's scores | **The collapse alarm.** Near zero means the probe outputs a constant, which converges nicely and distinguishes nothing. Guarded at 0.01 |
    | `val/target_mean` | Fraction of validation tokens that are positive | Fixed across recipes, because validation always uses the full split |
    | `val/prediction_mean` | Average score emitted | Far from `target_mean` means miscalibration, which the threshold step later absorbs |
    | `train/active_tokens` | Tokens actually reaching the loss, per forward pass | Times the accumulation, this is the realized token budget per step |
    | `train/pos_weight` | The class weight in use | Derived from the training stream's own balance |

    Three things worth knowing before comparing runs on these.

    **Every metric above exists once per depth.** A run carries a head at
    every layer, so each of these is logged as `val/layerNN/...`. The same
    name without a depth holds the *best depth at that step*, which is what
    the checkpoint is selected on, since the best depth is the probe the run
    would actually be used for. The depth selector above chooses which one
    the tables and figures below show; the aggregate appears beside it as
    `val/best_...` and belongs to no single layer.

    **The validation population is the same for every recipe.** Selection
    rules apply to training only; validation always scores every token of
    every validation rollout. So differences between runs are differences
    in the probe, not in what it was measured on.

    **Training positive rate and validation positive rate are meant to
    differ.** Composition deliberately over-samples positives in training,
    while validation reflects the real mix. That gap is why the class
    weight makes `val/loss` incomparable, and why selection runs on AUC.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading the numbers, part 2: the protocol

    The final numbers come from the black-box evaluator. It takes one score
    per token and knows nothing about what produced them.

    **First, the operating point.** A raw threshold is meaningless across
    scorers whose scores live on different scales. So instead a **budget**
    is fixed: the fraction of negative rollouts allowed to raise a false
    alarm, typically 1% and 5%. The threshold that spends exactly that
    budget is read off the validation negatives, then **frozen** and
    reused unchanged on test. Every number below is reported at a budget,
    never at a threshold.

    **Second, persistence.** A single token above threshold is usually
    noise. An alarm requires $m$ consecutive tokens above it. The first
    alarm position is

    $$a_r(\tau, m) = \min\{t : p_r(t') \ge \tau \ \text{for all} \ t' \in [t, t+m)\}$$

    and a rollout's score is the highest such persistent score it ever
    reaches.

    ### View A: detection

    Per rollout, at each budget.

    - **recall** — of the rollouts that really degenerate, the share caught
    - **precision** — of the rollouts flagged, the share that really degenerate
    - **negative_fpr** — realized false-alarm rate; should land on the budget
    - **rollout_auc / rollout_ap** — threshold-free ranking quality

    ### View B: coverage

    Per token, once the loop has begun.

    - **in_pattern_recall** — of the tokens at or after the frontier, the share flagged
    - **token_false_positive_rate** — of the tokens in negative rollouts, the share flagged

    A scorer can win View A and lose View B by firing once and going quiet.

    ### View C: lead time

    Per positive rollout, the distance from alarm to frontier.

    - **median_offset** — tokens between the alarm and the frontier.
      **Negative is early**, which is the useful direction; positive means
      the loop had already started
    - **never_fired_positives** — degenerate rollouts that were never
      flagged at all, which recall alone hides
    - **false_early_stop_rate** — how often acting on the alarm would have
      cut off a healthy generation

    ### View D: persistence

    How long the first alarm lasts, reported separately for the two
    populations, and **read in opposite directions**.

    - On **positives**, a long run is good: the alarm holds while the
      problem holds.
    - On **negatives**, every alarm is wrong, and the length separates a
      jittery scorer (short runs, which a larger $m$ would remove) from a
      confidently wrong one (long runs, which no $m$ can fix).

    ### What is never done

    Thresholds are never tuned on test. Per-domain cells backed by too few
    positives are marked anecdotal rather than reported as rates. Views are
    never read in isolation: a scorer that fires on every token has perfect
    recall, perfect coverage and perfect persistence, and only its
    false-alarm rate reveals it.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## The experiment register

    Every experiment is defined once, here, as data. The commands printed
    in each section are generated from this register, so a command shown in
    the notebook is the command that was run.
    """)
    return


@app.cell
def _():
    # A run carries a head at every depth. Reading the residual stream is what
    # a run spends its time on, and that cost is paid once whether one head or
    # thirty-one hang off it, so depth stops being a variable that multiplies
    # the number of runs and becomes an axis every result already has. Depth is
    # chosen when a result is read, never before it is measured, and it is a
    # scoring decision rather than a training one.
    PROBED_LAYERS = list(range(1, 32))
    # Every depth of every run is scored. Scoring a chosen few would be cheaper,
    # but choosing them means deciding in advance which depths a result could
    # possibly be read at, and that decision has to be made before the results
    # exist. The depth profile of one recipe is then also a measurement rather
    # than an assumption carried over from another.
    DEFAULT_LAYER = 12
    # The depth Stage 1 selects, and the one Stage 3 trains its single probe
    # at. Set it once the depth profile has been read; until then it is a
    # placeholder and Stage 3's commands are not ready to run.
    CHOSEN_DEPTH = DEFAULT_LAYER
    SEEDS = [42, 43, 44]

    BASE = {
        "training.features.regime": "cached",
        "training.lora.enabled": "false",
        # Equal budget: the same tokens per optimizer step and the same cap on
        # steps for every recipe, so the total gradient is held constant and
        # only the choice of tokens varies.
        #
        # The number is large enough that the widest window still fits a whole
        # micro-batch inside one step. A step cannot be smaller than one
        # micro-batch, so a budget below batch times window silently hands the
        # widest window more tokens than every recipe it is compared with.
        "training.budget.tokens_per_step": 4096,
        # A cap rather than a target. Training stops when every head has
        # stopped improving, and a run that reaches this number is a run whose
        # budget was too small rather than a run that finished.
        "training.runtime.max_steps": 2000,
        "training.runtime.per_device_train_batch_size": 8,
        # Often enough to see where a head stops improving. Most heads settle
        # inside the first few hundred steps, so a coarse cadence leaves the
        # only interesting stretch unmeasured and makes the earliest checkpoint
        # look best by default.
        "training.validation.strategy": "steps",
        "training.validation.steps": 50,
        "training.checkpoint.strategy": "steps",
        "training.checkpoint.steps": 50,
        # Every checkpoint is kept. Each holds one small head per depth, so a
        # long run costs a few hundred megabytes, and keeping them is what lets
        # a selection rule be reconsidered later without retraining anything.
        "training.checkpoint.save_total_limit": "null",
        # Monitoring reads a fixed subsample rather than the whole split, since
        # a probe covering many depths pays for every token of every layer each
        # time it is monitored.
        "training.validation.max_rollouts": 400,
        # A run writes no end-of-run evaluation. Scoring happens afterwards from
        # a checkpoint, one depth at a time.
        "training.validation.final_splits": "[]",
        "training.runtime.seed": 42,
        # Every setting any experiment varies is named here even when it equals
        # the configured default, so that a recipe is described the same way
        # wherever it appears and identical recipes collapse to one run.
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
    # each rollout contributes, which is a different question, so they are held
    # at one size instead of swept.
    POSITIONAL_RUNGS = ["random_window", "frontier_window", "frontier_window_hard_negative"]
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
        """Stage 1: which rule, and how large a window, decided together.

        Locking a window before the ladder runs requires the winner to transfer
        across rules, which is untested. Sweeping both at once costs more runs
        and answers the question that was actually asked.
        """
        runs = []
        for strategy in LADDER_RUNGS:
            sizes = WINDOWS if strategy in POSITIONAL_RUNGS else [128]
            for window in sizes:
                for seed in SEEDS:
                    label = f"{strategy}, W={window} (seed {seed})"
                    runs.append(
                        {
                            "label": label,
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
        """Stage 2a: does labelling the run-up positive make the alarm earlier?

        The horizon is a label axis, but its leverage depends on how much run-up
        the selection rule puts in front of the probe, so it is asked of three
        rules that differ in exactly that. Under `all_tokens` a horizon of 256
        adds about seven percent more positive tokens; under a centred window at
        512 the same horizon adds nearly eighty. If the axis is inert at both
        ends of that range, it is inert.

        A window has to be wide enough to express the horizon or the comparison
        measures the window instead: centred spends half its length after the
        frontier and so needs W at least twice the horizon, trailing is all
        run-up and needs W at least the horizon.
        """
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
                            "label": f"{strategy}/{anchor} W={window}, horizon {horizon} (seed {seed})",
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
        """Stage 2b: a target that decays back through the run-up.

        The decay length is bounded by the window for the same reason the
        horizon is: a centred window shows W/2 tokens of run-up, and a decay
        longer than that is a constant one over everything the probe sees.
        """
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
        """Stage 2c: a target defined everywhere, including on healthy text."""
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
        """Stage 2d: does correcting class imbalance twice change anything?

        Expected to be inert, because a threshold re-derived per run absorbs a
        calibration shift, which is most of what these two knobs do. Cheap
        enough to settle rather than assume.
        """
        return [
            {
                "label": f"pos_weight {'on' if weight else 'off'}, positives {fraction} (seed {seed})",
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
        """Stage 3: is the frozen representation the ceiling?

        The adapted regime runs the model, so it carries one probe at one depth
        rather than a head at every depth, and its control is a frozen run at
        that same depth. One head of a many-headed run would differ in more than
        the regime, since such a run selects on whichever depth leads and fits
        one class weight for all of them.

        A randomly initialised probe pushes gradients into the adapters before
        it knows what it is looking for, which corrupts the representation while
        the probe is still noise. The adapters therefore move an order of
        magnitude more slowly than the head. Freezing them outright for the
        first few hundred steps is the stronger version of the same fix and is
        not implemented; if the slower rate is not enough, that is the next
        thing to build rather than a reason to accept the result.

        The depth is the one Stage 1 selects, which is a number this notebook
        cannot know in advance. It is held in CHOSEN_DEPTH and is the single
        edit Stage 3 needs.
        """
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
            "title": "Baselines through the same protocol",
            "depends_on": [],
            "runs": [],
            "shell": ["sbatch cluster/baselines.sbatch"],
        },
        {
            "id": "S1",
            "title": "Which rule, and how large a window",
            "depends_on": [],
            "runs": grid_runs,
            "shell": [],
        },
        {
            "id": "S2a",
            "title": "Does the horizon buy lead time?",
            "depends_on": [],
            "runs": horizon_runs,
            "shell": [],
        },
        {
            "id": "S2b",
            "title": "Soft labels",
            "depends_on": ["S1"],
            "runs": soft_runs,
            "shell": [],
        },
        {
            "id": "S2c",
            "title": "Regression instead of detection",
            "depends_on": ["S1"],
            "runs": regression_runs,
            "shell": [],
        },
        {
            "id": "S2d",
            "title": "Class balance and calibration",
            "depends_on": ["S1"],
            "runs": balance_runs,
            "shell": [],
        },
        {
            "id": "S3",
            "title": "Do adapters earn their cost?",
            "depends_on": ["S1", "S2a", "S2b", "S2c", "S2d"],
            "runs": adapted_runs,
            "shell": [],
        },
        {
            "id": "S4",
            "title": "The held-out test, once",
            "depends_on": ["S3"],
            "runs": [],
            "shell": [],
        },
    ]
    BY_ID = {entry["id"]: entry for entry in EXPERIMENTS}
    return BY_ID, EXPERIMENTS


@app.cell
def _(EXPERIMENTS, mo, shlex):
    # Several experiments ask different questions of the same configuration:
    # the window pilot's W=128 run is also the ladder's third rung at seed 42,
    # and the layer sweep's layer 30 is the ladder's fourth. A run's identity
    # comes from its settings, so those are one run, not several, and
    # submitting them separately would put two jobs in one directory.
    #
    # Each distinct configuration is therefore planned once and carries the tag
    # of every experiment it belongs to.
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
        overrides = [f"{key}={value}" for key, value in sorted(slot["overrides"].items())]
        overrides.append(f"training.wandb.tags=[{tags}]")
        rendered = " ".join(shlex.quote(item) for item in overrides)
        # A recipe may need more than the queue script's default walltime.
        flags = f"{slot['sbatch']} " if slot.get("sbatch") else ""
        return f"sbatch {flags}cluster/train.sbatch {rendered}"

    def commands_for(exp_id):
        entry = next(e for e in EXPERIMENTS if e["id"] == exp_id)
        lines = list(entry["shell"])
        lines += [render(slot) for slot in PLAN if exp_id in slot["exps"]]
        return lines

    def shared_note(exp_id):
        shared = [
            slot for slot in PLAN if exp_id in slot["exps"] and len(slot["exps"]) > 1
        ]
        if not shared:
            return None
        others = sorted({e for slot in shared for e in slot["exps"]} - {exp_id})
        return mo.md(
            f"_{len(shared)} of these runs are shared with {', '.join(others)}: "
            "the same configuration answers more than one question, so it is "
            "trained once and tagged for each._"
        )

    def show_commands(exp_id):
        lines = commands_for(exp_id)
        if not lines:
            return mo.md("_No commands yet: this experiment depends on earlier results._")
        body = "\n".join(lines)
        return mo.accordion(
            {
                f"Reproduce ({len(lines)} commands)": mo.md(
                    f"```bash\ncd /iopsstor/scratch/cscs/mdenegri/degeneration-probe\n{body}\n```"
                )
            }
        )

    return PLAN, commands_for, render, shared_note, show_commands


@app.cell
def _(OUTPUTS, json, mo, pd):
    def _checkpoint_step(path):
        if not path:
            return None
        tail = str(path).rsplit("checkpoint-", 1)
        return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None

    def load_all_runs():
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
            rows.append(
                {
                    # A run can belong to several experiments, so this is a
                    # list and is matched by membership rather than equality.
                    "exps": [t.split(":", 1)[1] for t in tags if t.startswith("exp:")],
                    "group": info.get("group"),
                    "status": info.get("status"),
                    "minutes": round((info.get("duration_seconds") or 0) / 60, 1),
                    "steps": training.get("global_step"),
                    "epochs": training.get("epochs_completed"),
                    "best": training.get("best_metric"),
                    "selected_on": training.get("metric_for_best_model"),
                    # The step the reported probe was taken from, which is not
                    # the step the run reached.
                    "selected_step": _checkpoint_step(training.get("best_checkpoint")),
                    "tokens_per_step": (training.get("budget") or {}).get(
                        "tokens_per_step_realized"
                    ),
                    "tokens_per_example": (training.get("budget") or {}).get(
                        "tokens_per_example"
                    ),
                    "seed": axes.get("seed"),
                    "layer": axes.get("layer"),
                    "selection": axes.get("selection"),
                    "window": axes.get("window"),
                    "anchor": axes.get("anchor"),
                    "label": axes.get("label"),
                    "horizon": axes.get("horizon"),
                    "loss": axes.get("loss"),
                    "pos_weight": axes.get("pos_weight"),
                    "features": axes.get("features"),
                    "run_dir": str(info_path.parent),
                    "commit": (info.get("environment", {}).get("git_commit") or "")[:8],
                }
            )
        return pd.DataFrame(rows)

    def missing(message):
        return mo.md(f"**Nothing to show yet.** {message}").callout(kind="warn")

    all_runs = load_all_runs()
    return all_runs, missing


@app.cell
def _(BY_ID, all_runs, commands_for, missing, mo, pd, shared_note):
    def tagged_with(exp_id):
        if all_runs.empty or "exps" not in all_runs.columns:
            return all_runs
        return all_runs[all_runs["exps"].apply(lambda tags: exp_id in tags)]

    def run_status(exp_id):
        expected = len(BY_ID[exp_id]["runs"]) or len(commands_for(exp_id))
        if all_runs.empty or "exps" not in all_runs.columns:
            return missing(
                f"No runs found under `outputs/`. Expected {expected} for {exp_id}; "
                "submit the commands above."
            )
        found = tagged_with(exp_id)
        if found.empty:
            return missing(
                f"No run is tagged `exp:{exp_id}` yet. Expected {expected}; "
                "submit the commands above."
            )
        done = int((found["status"] == "finished").sum())
        header = mo.md(
            f"**{done} of {expected} finished** "
            f"({len(found) - done} running, failed, or interrupted)."
        )
        note = shared_note(exp_id)
        columns = [
            c
            for c in (
                "status", "minutes", "steps", "best", "seed", "layer", "selection",
                "window", "anchor", "label", "horizon", "loss", "pos_weight", "run_dir",
            )
            if c in found.columns
        ]
        table = pd.DataFrame(found[columns]).reset_index(drop=True)
        return mo.vstack([header, note, table] if note is not None else [header, table])

    return run_status, tagged_with


@app.cell
def _(Path, missing, pd, re, tagged_with):
    # A run holds a probe at every depth, so depth is a column here rather than
    # something that distinguishes one run directory from another. Both readers
    # below turn what is stored per depth into a `layer` column, so that a
    # single-depth run and a run covering all of them read the same way and can
    # be put on one axis.
    LAYER_COLUMN = re.compile(r"^(?P<prefix>.*?)layer(?P<layer>\d+)/(?P<metric>.+)$")

    def run_rows_for(exp_id):
        found = tagged_with(exp_id)
        if found.empty or "status" not in found.columns:
            return []
        finished = found[found["status"] == "finished"]
        return list(finished[["run_dir", "layer"]].itertuples(index=False, name=None))

    def run_dirs_for(exp_id):
        return [run_dir for run_dir, _ in run_rows_for(exp_id)]

    def _spread_layers(frame, own_layer):
        """Turn `val/layerNN/metric` columns into rows carrying a layer."""
        found = {}
        for column in frame.columns:
            match = LAYER_COLUMN.match(column)
            if match:
                renamed = match.group("prefix") + match.group("metric")
                found.setdefault(int(match.group("layer")), {})[column] = renamed
        if not found:
            # A run that trained one depth already has the plain names; it only
            # needs to say which depth they belong to.
            plain = frame.copy()
            plain["layer"] = own_layer
            return plain
        shared = [c for c in frame.columns if not LAYER_COLUMN.match(c)]
        # Beside the per-depth columns, a run also logs each metric without a
        # depth, holding the best depth at that step. That is what the
        # checkpoint is selected on, and it belongs to no single layer, so it
        # keeps a name of its own rather than colliding with the depth it
        # happened to come from.
        wanted = {new for renames in found.values() for new in renames.values()}
        best = {}
        for column in shared:
            if column in wanted:
                prefix, _, metric = column.rpartition("/")
                best[column] = f"{prefix}/best_{metric}" if prefix else f"best_{metric}"
        pieces = []
        for layer, renames in sorted(found.items()):
            piece = frame[shared + list(renames)].rename(columns={**best, **renames})
            piece["layer"] = layer
            pieces.append(piece)
        return pd.concat(pieces, ignore_index=True)

    def load_curves(exp_id, layer=None):
        frames = []
        for run_dir, own_layer in run_rows_for(exp_id):
            path = Path(run_dir) / "history.parquet"
            if not path.is_file():
                continue
            frame = _spread_layers(pd.read_parquet(path), own_layer)
            frame["run"] = Path(run_dir).parent.name
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        curves = pd.concat(frames, ignore_index=True)
        if layer is not None:
            curves = curves[curves["layer"] == layer].reset_index(drop=True)
        return curves

    def _evaluation_dir(run_dir, own_layer, split, layer):
        """Where one depth of one run keeps its protocol output, if it has any."""
        root = Path(run_dir)
        if layer is not None:
            scoped = root / "layers" / f"layer_{int(layer):02d}" / "evaluation" / split
            if scoped.is_dir():
                return scoped
            # A run that trained a single depth keeps its output at the root,
            # but it can only answer for the depth it actually trained.
            if own_layer is not None and int(own_layer) != int(layer):
                return None
        plain = root / "evaluation" / split
        return plain if plain.is_dir() else None

    def scored_layers(exp_id, split="val"):
        """Which depths of this experiment have been through the evaluator."""
        depths = set()
        for run_dir, own_layer in run_rows_for(exp_id):
            root = Path(run_dir)
            for scoped in sorted(root.glob("layers/layer_*/evaluation")):
                if (scoped / split).is_dir():
                    depths.add(int(scoped.parent.name.split("_")[-1]))
            if (root / "evaluation" / split).is_dir() and own_layer is not None:
                depths.add(int(own_layer))
        return sorted(depths)

    def load_views(exp_id, split, view, layer=None):
        frames = []
        for run_dir, own_layer in run_rows_for(exp_id):
            directory = _evaluation_dir(run_dir, own_layer, split, layer)
            if directory is None or not (directory / f"{view}.csv").is_file():
                continue
            frame = pd.read_csv(directory / f"{view}.csv")
            frame.insert(0, "run", Path(run_dir).parent.name)
            frame.insert(1, "layer", layer if layer is not None else own_layer)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def protocol_table(exp_id, split, layer=None):
        detection = load_views(exp_id, split, "view_a_detection", layer)
        if detection.empty:
            available = scored_layers(exp_id, split)
            where = (
                f" Depths scored so far: {available}."
                if available
                else " No depth of this experiment has been scored yet."
            )
            return missing(
                f"No run for {exp_id} has protocol output on `{split}` at layer "
                f"{layer}.{where} Training writes no scores by itself: run "
                "`cluster/score_layers.sbatch <run_dir> \"8 12 30\" <split>` for "
                "each finished run, which scores those depths and puts each "
                "through the evaluator."
            )
        merged = detection
        for view, columns in (
            ("view_c_lead_time", ["median_offset", "never_fired_positives", "false_early_stop_rate"]),
            ("view_b_coverage", ["in_pattern_recall", "token_false_positive_rate"]),
        ):
            extra = load_views(exp_id, split, view, layer)
            if extra.empty:
                continue
            keep = ["run", "layer", "target_negative_fpr"] + [
                c for c in columns if c in extra.columns
            ]
            merged = merged.merge(
                extra[keep], on=["run", "layer", "target_negative_fpr"], how="left"
            )
        return merged

    def depth_table(exp_id, split, budget=0.01, layers=None):
        """One row per run and depth, for reading a result along its depth axis."""
        depths = layers if layers is not None else scored_layers(exp_id, split)
        frames = []
        for layer in depths:
            table = protocol_table(exp_id, split, layer)
            if hasattr(table, "columns") and not table.empty:
                frames.append(table[table["target_negative_fpr"] == budget])
        if not frames:
            return missing(
                f"No depth of {exp_id} has protocol output on `{split}` yet."
            )
        return pd.concat(frames, ignore_index=True).sort_values(["layer", "run"])

    return load_curves, protocol_table, run_dirs_for


@app.cell
def _(FIGURES, load_curves, missing, plt):
    def curve_figure(exp_id, layer=None):
        frame = load_curves(exp_id, layer)
        if frame.empty:
            return missing(
                f"No finished run for {exp_id} has written `history.parquet` "
                f"at layer {layer}."
            )
        columns = [
            ("loss", "training loss"),
            ("val/loss_unweighted", "validation loss (unweighted)"),
            ("val/recall_at_budget", "recall at a 1% false-alarm budget"),
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
                    axis.plot(points["step"], points[column], marker="o", ms=3, label=run[:38])
            if column == "val/prediction_std":
                axis.axhline(0.01, color="crimson", ls="--", lw=1)
            axis.set_title(title)
            axis.set_xlabel("step")
            axis.grid(alpha=0.3)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7)
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        suffix = "" if layer is None else f"_L{int(layer):02d}"
        path = FIGURES / f"{exp_id}{suffix}_curves"
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        return fig

    return (curve_figure,)


@app.cell
def _(FIGURES, missing, plt, protocol_table):
    def operating_figure(exp_id, split, layer=None):
        frame = protocol_table(exp_id, split, layer)
        if not hasattr(frame, "columns"):
            return frame
        if frame.empty:
            return missing(f"No protocol output for {exp_id} on `{split}` at layer {layer}.")
        panels = [
            ("recall", "recall"),
            ("precision", "precision"),
            ("median_offset", "median lead time (below zero is early)"),
        ]
        available = [(c, title) for c, title in panels if c in frame.columns]
        fig, axes = plt.subplots(1, len(available), figsize=(4.6 * len(available), 3.8))
        axes = axes if hasattr(axes, "__len__") else [axes]
        for axis, (column, title) in zip(axes, available):
            for run, group in frame.groupby("run"):
                points = group.sort_values("target_negative_fpr")
                axis.plot(
                    points["target_negative_fpr"],
                    points[column],
                    marker="o",
                    label=run[:34],
                )
            if column == "median_offset":
                axis.axhline(0, color="black", lw=1)
            axis.set_xlabel("false-alarm budget")
            axis.set_title(title)
            axis.grid(alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7)
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        suffix = "" if layer is None else f"_L{int(layer):02d}"
        path = FIGURES / f"{exp_id}_{split}{suffix}_operating"
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        return fig

    return (operating_figure,)


@app.cell
def _(mo):
    mo.md(r"""
    ### What the equal budget selects

    Every recipe asks for the same number of tokens per optimizer step and
    runs for the same number of steps, so no recipe can buy an advantage
    with a longer run. Two things break the symmetry anyway, and both are
    measured rather than assumed.

    **The step budget is requested, not granted.** Accumulation is sized
    from the tokens an example really contributes, and it has to be a whole
    number, so a recipe whose micro-batch already sits near the budget
    rounds to one accumulated batch and takes whatever that batch holds.
    A wide window is the case that suffers: its windows are clipped by the
    ends of the rollouts they sit in, so it carries fewer tokens than its
    width suggests and lands furthest below the request. The table reports
    the realized figure against the requested one, and a comparison across
    windows is only as good as the gap between them is small.

    **The reported probe is not the trained probe.** A run trains to the
    cap, but what is carried forward is the checkpoint the selection metric
    chose, and every step after it is discarded. So the training that
    reaches the results is the prefix up to that checkpoint, which differs
    between recipes and can be a small fraction of the run.

    The pools differ as sharply: a rule with a large pool never finishes a
    pass over it, while a rule with a small pool goes round several times.

    That raises the obvious question, which is what decides *which* tokens
    a rule uses when it only gets through a fraction of its pool. Three
    stages answer it, and only the last one truncates.

    1. **Tiling or drawing.** The rule turns rollouts into candidate
       windows. Nothing is discarded yet.
    2. **Composition fixes the order.** Positive windows are shuffled
       across the whole split, then consumed a fixed number per batch;
       negative windows are drawn per batch in proportion to each domain's
       share. This is what holds every batch at the same class mix.
    3. **Training reads that order in sequence**, and the selected
       checkpoint takes a prefix of it.

    Because the shuffle happens in stage 2 and the cut in stage 3, **a
    prefix is a uniform random sample**, not the first rollouts in file
    order. A rule that gets through a tenth of its pool still draws that
    tenth from everywhere, across every domain and every position within a
    rollout.

    The table below is read from the finished runs rather than derived
    from the settings, because the two disagree. It reports the tokens a
    step really saw, how far the run trained, and how far it had got when
    the checkpoint that gets reported was written. That last column is the
    one that counts: a run trains to the cap, but the probe carried
    forward is the selected checkpoint, and everything after it is
    discarded.

    The column to read alongside the unique-window count is the number of
    *rollouts* touched: a rule can use a small share of its windows while
    still seeing nearly every degeneration episode, which is a very
    different situation from having seen only a small share of the
    episodes.
    """)
    return


@app.cell
def _(DEFAULT_LAYER, REPO, all_runs, missing, mo, pd):
    def budget_coverage():
        import numpy as np
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from degeneration_probe.config import (
            ExperimentConfig,
            LabelConfig,
            SelectionConfig,
        )
        from degeneration_probe.data.dataset import (
            load_degeneration_records,
            load_token_signal,
        )
        from degeneration_probe.data.windowed_dataset import WindowedActivationDataset

        if all_runs.empty:
            return missing("No runs have been written yet.")
        done = all_runs[
            (all_runs["status"] == "finished") & all_runs["selected_step"].notna()
        ]
        if done.empty:
            return missing("No run has finished, so nothing has been consumed yet.")

        try:
            with initialize_config_dir(
                config_dir=str(REPO / "configs"), version_base=None
            ):
                composed = compose(config_name="main")
            experiment = ExperimentConfig.from_dict(
                OmegaConf.to_container(composed, resolve=True)
            )
            records = load_degeneration_records(
                experiment.dataset,
                split="train",
                label_config=LabelConfig(),
                training=True,
            )
        except Exception as error:
            return missing(f"Could not read the training split: {error}")

        defaults = OmegaConf.to_container(composed.training.selection, resolve=True)
        positive_rollouts = sum(record.is_positive for record in records)
        # Hard-negative placement reads where a rollout looks repetitive, which
        # the other rules never ask for.
        negatives = [
            (index, record)
            for index, record in enumerate(records)
            if not record.is_positive
        ]
        hardness = {
            negatives[position][0]: values
            for position, values in load_token_signal(
                experiment.dataset, [record for _, record in negatives]
            ).items()
        }

        rows = []
        for (rule, window), group in done.groupby(
            ["selection", "window"], dropna=False
        ):
            selection = {**defaults, "strategy": rule}
            if pd.notna(window):
                selection.update(window_size=int(window), anchor="centered")
            dataset = WindowedActivationDataset(
                records,
                build_root=experiment.dataset.build_root,
                # Which depth is read does not change which windows the rule
                # builds, and this table is about the windows.
                probe_layer=DEFAULT_LAYER,
                selection=SelectionConfig(**selection),
                batch_size=8,
                seed=42,
                hardness=hardness
                if rule == "frontier_window_hard_negative"
                else None,
            )
            order = np.array(dataset.order)
            is_positive = np.array(
                [
                    dataset.records[dataset.windows[index].record_index].is_positive
                    for index in order
                ]
            )

            # A run's length in passes is recorded; the prefix that reached the
            # selected checkpoint is that length scaled by where it sat.
            passes = float(group["epochs"].median())
            fraction = float(
                (group["selected_step"] / group["steps"]).median()
            )
            consumed = int(round(passes * fraction * len(order)))
            seen = order[: min(consumed, len(order))]
            seen_positive = seen[is_positive[: len(seen)]]
            rows.append(
                {
                    "rule": rule,
                    "window": None if pd.isna(window) else int(window),
                    "runs": len(group),
                    "tokens per step": int(group["tokens_per_step"].mean()),
                    "of requested": f"{100 * group['tokens_per_step'].mean() / 4096:.0f}%",
                    "slots in one pass": len(order),
                    "trained to": int(group["steps"].median()),
                    "passes trained": round(passes, 2),
                    "selected at": int(group["selected_step"].median()),
                    "slots consumed": min(consumed, len(order)),
                    "passes consumed": round(passes * fraction, 3),
                    "positive windows used": len(set(seen_positive.tolist())),
                    "positive windows available": int(is_positive.sum()),
                    "positive rollouts touched": len(
                        {dataset.windows[int(i)].record_index for i in seen_positive}
                    ),
                    "of positive rollouts": positive_rollouts,
                }
            )
        frame = pd.DataFrame(rows).sort_values(["rule", "window"], na_position="first")
        return mo.ui.table(frame, selection=None)

    budget_coverage()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Choosing the checkpoint after the fact

    Selection during a run reads one number, and that number has to be
    picked before any of this is known. Every checkpoint is kept and every
    evaluation is recorded per depth, so the choice does not have to stand:
    a rule can be applied to the recorded history afterwards and the
    checkpoint it names loaded, without retraining anything.

    Two limits are worth stating. Only what the monitor recorded can be
    re-selected on, which is the rollout-level view and the loss; coverage
    and distance from the frontier are not in it and need the scoring
    pipeline run over checkpoints. And the monitor is a thinned split, so
    what it supports is choosing a checkpoint, never reporting a result.

    The table names, for one depth, the step each candidate rule would
    choose. Where the rules disagree, the number carried into the results
    is a consequence of the rule and not of the run.
    """)
    return


@app.cell
def _(Path, pd, re):
    DEPTH_METRIC = re.compile(r"^val/layer(?P<layer>\d+)/(?P<metric>.+)$")

    # Whether a larger value is better, for every metric worth selecting on.
    DIRECTION = {
        "recall_at_budget": True,
        "rollout_auc": True,
        "rollout_ap": True,
        "loss": False,
        "loss_unweighted": False,
    }

    def depth_curves(run_dir):
        """One row per (step, depth) of what validation recorded."""
        path = Path(run_dir) / "history.parquet"
        if not path.is_file():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        pieces = []
        for column in frame.columns:
            match = DEPTH_METRIC.match(column)
            if match is None:
                continue
            pieces.append(
                pd.DataFrame(
                    {
                        "step": frame["step"],
                        "layer": int(match.group("layer")),
                        "metric": match.group("metric"),
                        "value": frame[column],
                    }
                )
            )
        if not pieces:
            return pd.DataFrame()
        tidy = pd.concat(pieces, ignore_index=True).dropna(subset=["value"])
        return (
            tidy.pivot_table(index=["step", "layer"], columns="metric", values="value")
            .reset_index()
            .rename_axis(columns=None)
        )

    def select_step(curves, metric, layer):
        """The checkpoint a rule names for one depth. Ties go to the earliest.

        Earliest rather than latest because a tie means the rule cannot tell
        the two apart, and the cheaper of two indistinguishable checkpoints is
        the one that spent less to get there.
        """
        if curves.empty or metric not in curves.columns:
            return None
        one = curves[curves["layer"] == layer].sort_values("step")
        one = one.dropna(subset=[metric])
        if one.empty:
            return None
        target = one[metric].max() if DIRECTION[metric] else one[metric].min()
        return int(one.loc[one[metric] == target, "step"].iloc[0])

    def selection_spread(curves, metric, layers=None):
        """Where each depth peaks, which a single global choice has to ignore."""
        if curves.empty or metric not in curves.columns:
            return pd.DataFrame()
        layers = sorted(curves["layer"].unique()) if layers is None else layers
        return pd.DataFrame(
            [
                {
                    "layer": layer,
                    "selects at": select_step(curves, metric, layer),
                    "value there": round(
                        float(
                            curves[
                                (curves["layer"] == layer)
                                & (curves["step"] == select_step(curves, metric, layer))
                            ][metric].iloc[0]
                        ),
                        4,
                    ),
                    "distinct values": curves[curves["layer"] == layer][metric].nunique(),
                }
                for layer in layers
                if select_step(curves, metric, layer) is not None
            ]
        )

    return DIRECTION, depth_curves, select_step, selection_spread


@app.cell
def _(DEFAULT_LAYER, DIRECTION, all_runs, depth_curves, missing, mo, pd, select_step):
    def reselection_table(layer=DEFAULT_LAYER):
        if all_runs.empty:
            return missing("No runs have been written yet.")
        done = all_runs[all_runs["status"] == "finished"]
        if done.empty:
            return missing("No run has finished.")
        rows = []
        for run in done.itertuples():
            curves = depth_curves(run.run_dir)
            if curves.empty:
                continue
            row = {
                "rule": run.selection,
                "window": run.window,
                "seed": run.seed,
                "trained to": run.steps,
                "recorded": run.selected_step,
            }
            for metric in DIRECTION:
                row[metric] = select_step(curves, metric, layer)
            rows.append(row)
        if not rows:
            return missing("No finished run has written a history yet.")
        frame = pd.DataFrame(rows)
        return mo.ui.table(
            frame.sort_values(["rule", "window", "seed"], na_position="first"),
            selection=None,
        )

    reselection_table()
    return (reselection_table,)


@app.cell
def _(mo):
    mo.md("""
    Two things follow from this that are worth checking rather than
    assuming.

    **Redrawing versus repeating.** A rule that completes several passes
    redraws its windows at each pass boundary, so it does not see the same
    tokens again; it sees freshly placed ones. Comparing rules on windows
    seen *per rollout* rather than on total windows is therefore the fair
    reading, and by that measure the rules are much closer than the
    raw counts suggest.

    **A rule that never completes a pass never redraws.** Its sample is
    fixed for the whole run while others keep drawing. The effect should be
    small at these numbers, but it is real and it is not part of what the
    ladder sets out to measure.

    The budget is only fair if it is not also starving anyone, and the
    check is the training curve. A selection metric still climbing at the
    step limit means the run was cut short rather than fairly constrained,
    and the budget needs raising for every recipe together. A metric that
    reached its ceiling in the first few evaluations is the opposite
    failure and the more dangerous one: nothing after that point can be
    told apart, the selected checkpoint is whichever early evaluation
    happened to touch the ceiling first, and the comparison is then between
    arbitrary early probes rather than between recipes.
    """)
    return


@app.cell
def _(DEFAULT_LAYER, PROBED_LAYERS, mo):
    split_choice = mo.ui.dropdown(
        options=["val", "test_indomain", "test_heldout_domains"],
        value="val",
        label="Split shown in every experiment below",
    )
    # Depth is an axis of every result rather than a property of a run, so it
    # is chosen here, when a result is read. Every depth of every scored run is
    # available, and a run that has not been scored yet shows a message rather
    # than an empty table.
    # A mapping rather than a list, so the value read downstream is the integer
    # depth and not the label that stands for it.
    layer_choice = mo.ui.dropdown(
        options={str(n): n for n in PROBED_LAYERS},
        value=str(DEFAULT_LAYER),
        label="Depth shown in every experiment below",
    )
    mo.hstack([split_choice, layer_choice], justify="start", gap=2)
    return layer_choice, split_choice


@app.cell
def _(mo):
    mo.md("""
    ---
    # How the stages fit together

    Four stages, run in order, each ending in a decision the next one needs.
    Within a stage the runs are independent of one another and can all be
    queued at once.

    | stage | question | runs |
    |---|---|---|
    | **S0** | do model-free signals already do this? | none, one job |
    | **S1** | which selection rule, and how large a window? | 42 |
    | **S2a–d** | which target? | 54 |
    | **S3** | is the frozen representation the ceiling? | 6 |
    | **S4** | how does the winner do on held-out data? | none, scoring only |

    **What every run shares.** A head at every depth, a token budget of 2048
    per step, validation and a checkpoint every 50 steps with every checkpoint
    kept, and a cap of 2000 steps that training is not expected to reach.
    Three seeds wherever a claim ranks one recipe above another.

    **What stops a run.** Not the step cap. Each head is watched separately and
    frozen once its token coverage stops improving; the run ends when the last
    head is done. The checkpoint kept for a head is the one nearest the
    frontier among those within a tolerance of its best coverage.

    Coverage decides *when to stop* because it is an average over tens of
    thousands of tokens and moves smoothly. Distance from the frontier decides
    *which checkpoint to keep* because it is what the probe is for. Using the
    second to detect a plateau would stop runs on noise: it is a median over
    roughly a hundred rollouts and swings by more between neighbouring
    checkpoints than it moves over all of training.

    **What is never tuned here.** The test splits are not read until S4.
    Thresholds are frozen on validation and reused unchanged.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S0. Baselines through the same protocol

    **The question.** How much of this can be done without a probe at all?

    Three model-free scorers go through the identical evaluator: a repetition
    score over a sliding window, the longest repeated substring, and the
    model's own predictive entropy. They see the same rollouts, get thresholds
    frozen the same way, and are reported in the same four views.

    **What to expect.** Rollout-level detection is the easy question, so a
    simple repetition counter should do well on it. The interesting comparison
    is token coverage and distance from the frontier: a counter can only fire
    once repetition has already happened, so it has no way to be early. A probe
    that cannot beat it on those has not earned its cost.

    **Why this runs first.** It depends on nothing, it is one job, and every
    later number wants something to be measured against.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("S0")
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("S0", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S1. Which rule, and how large a window

    **The question.** Two, deliberately asked together.

    The five rungs each change exactly one decision about *which tokens the
    probe learns from*, from the whole corpus down to windows placed on the
    frontier and biased toward confusable negatives. The window size says how
    much context one of those windows carries.

    These are not independent. An anchored window trades coverage for
    earliness, and its size moves the probe along that same trade-off, so
    picking a size first and locking it assumes the winner transfers across
    rules. That assumption is untested, and asking both at once costs 42 runs
    instead of 15 and answers the question that was actually asked.

    Window size is swept only for the three rules that place a window
    deliberately. For `all_tokens` it sets how tokens are tiled and for
    `rollout_balanced` how many each rollout contributes, which are different
    questions and not this one.

    **What to expect.** Rollout-level recall will not separate the rungs: every
    rule catches nearly every degenerate rollout, so that view saturates. The
    separation, if there is one, shows up in token coverage and in distance
    from the frontier, and it is those two the stage is read on.

    **Scoring.** Every depth of every run, on validation. A depth costs about
    three minutes, so a run is about an hour and a half and the stage is roughly
    seventy GPU-hours, which parallelises down to an afternoon.

    Scoring a chosen few would be four times cheaper and buys a decision nobody
    wants to make: which depths a result could possibly be read at, decided
    before the results exist. It also makes the depth profile of every rule a
    measurement rather than one rule's profile assumed to hold for the others.
    Whether the best depth depends on the selection rule is then something this
    stage answers rather than something it takes on trust.

    Depth is never a training choice. Every run already carries every depth;
    scoring only decides which of them can be read.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("S1")
    return


@app.cell
def _(mo):
    mo.md("""
    **Scoring command**, once the runs have finished. One job per run, and they
    are all independent of one another:

    ```bash
    for run in outputs/*/2*/; do
        sbatch --time=03:00:00 cluster/score_layers.sbatch         "$(realpath $run)" "$(seq -s' ' 1 31)" val
    done
    ```
    """)
    return


@app.cell
def _(run_status):
    run_status("S1")
    return


@app.cell
def _(curve_figure, layer_choice):
    curve_figure("S1", layer_choice.value)
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("S1", split_choice.value, layer_choice.value)
    return


@app.cell
def _(layer_choice, operating_figure, split_choice):
    operating_figure("S1", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S2a. Does the horizon buy lead time?

    **The question.** The hard frontier label calls a token one position before
    the loop as innocent as one a thousand positions before. The horizon moves
    that boundary earlier, asking the probe to fire while the text still looks
    fine. Whether that produces an earlier alarm is the whole of its claim.

    **Why it is asked of three rules.** The horizon is a property of the label,
    but its leverage depends on how much run-up the selection rule puts in
    front of the probe. Under `all_tokens` a horizon of 256 adds about seven
    percent more positive tokens, because onsets land early and most of a
    degenerate rollout is already loop. Under a centred window at 512 the same
    horizon adds nearly eighty percent. A trailing window is all run-up, so the
    horizon controls its entire positive class.

    Asking only where leverage is high would leave the interaction untested,
    and asking only where it is low would predict a null from the label counts
    alone. Both ends are therefore in.

    **The constraint that makes this measurable.** A window has to be wide
    enough to express the horizon. A centred window spends half its length past
    the frontier, so it needs a width of at least twice the horizon; a trailing
    window needs at least the horizon. Below that, two different horizons label
    every token in the window positive, train on identical data, and get
    reported as two points that differ in nothing.

    **What to expect.** A previous single-seed attempt found the median alarm
    moving by a couple of tokens across an eightfold range of horizons, which
    is under the seed-to-seed spread and so cannot be distinguished from no
    effect. Three seeds settles that. This stage is also read on **distance**
    from the frontier rather than signed offset: if the frontier marks where
    degeneration begins, firing five hundred tokens early is a false alarm
    inside a positive rollout rather than an achievement.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("S2a")
    return


@app.cell
def _(run_status):
    run_status("S2a")
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("S2a", split_choice.value, layer_choice.value)
    return


@app.cell
def _(layer_choice, operating_figure, split_choice):
    operating_figure("S2a", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S2b. Soft labels

    **The question.** The same frontier with the step replaced by a decay, so
    that closer to the loop means more degenerate without committing to a hard
    cut. The decay length says how far back the run-up is taken to reach.

    **The constraint.** The same bound as the horizon, with the decay length in
    its place. A centred window shows half its width of run-up, so a decay
    longer than that is a constant one over everything the probe sees. These
    run at a width of 512 with decay lengths of 128 and 256 for that reason.

    **What to expect.** The class weight is off here, because a soft target has
    no class to weight, which also means the loss is not comparable with the
    hard-label runs and only the protocol views are.

    If the earlier finding holds that anchored rules win by reading the
    approach better rather than by reading the loop differently, then a target
    that grades the approach is aimed at the same mechanism from the label side
    and is the most likely of the target axes to move something.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("S2b")
    return


@app.cell
def _(run_status):
    run_status("S2b")
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("S2b", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S2c. Regression instead of detection

    **The question.** A target that exists everywhere, including on healthy
    rollouts, where a nonzero repetition score describes text that is genuinely
    repetitive and perfectly legitimate.

    This trains a different concept from "this rollout has broken", which is
    exactly why it is an axis and not a foregone conclusion. A probe that
    tracks repetition will fire on a numbered list; a probe that tracks
    degeneration should not.

    **What to expect.** Better token coverage inside a loop, because the target
    is dense there, and worse false-alarm behaviour on legitimate repetition.
    The comparison to watch is the token-level false-positive rate at a fixed
    budget, not recall.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("S2c")
    return


@app.cell
def _(run_status):
    run_status("S2c")
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("S2c", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S2d. Class balance and calibration

    **The question.** Imbalance can be corrected in the composition of the
    training stream or in the loss, and doing both at once corrects it twice.
    This crosses the class weight with the fraction of positive windows.

    **What to expect.** Very likely nothing. Both knobs mostly shift where the
    scores sit rather than how well they separate, and a threshold re-derived
    per run absorbs a shift. The reason to run it anyway is that "very likely
    nothing" is a prediction, it costs twelve runs, and leaving it unmeasured
    means every later result carries an untested assumption.

    The number that would change the verdict is the score spread: a probe whose
    scores collapse toward a constant converges nicely and distinguishes
    nothing, and that failure is invisible in the loss.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("S2d")
    return


@app.cell
def _(run_status):
    run_status("S2d")
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("S2d", split_choice.value, layer_choice.value)
    return


@app.cell
def _(Path, json, missing, mo, pd, run_dirs_for):
    def composition_table(exp_id):
        """What the probe actually saw, split by split, as it was assembled.

        Composition and the class weight both correct imbalance, so the pair is
        read together: the realized positive rate of the training stream beside
        the weight in force is what makes a double correction visible.
        """
        rows = []
        for run_dir in run_dirs_for(exp_id):
            path = Path(run_dir) / "dataset_summary.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text())
            for split, values in summary.items():
                if not isinstance(values, dict):
                    continue
                rows.append(
                    {
                        "run": Path(run_dir).parent.name[:46],
                        "split": split,
                        "examples": values.get("windows") or values.get("rollouts"),
                        "valid_tokens": values.get("valid_tokens"),
                        "positive_token_rate": round(values.get("positive_token_rate", 0), 4),
                        "pos_weight": summary.get("pos_weight"),
                        "effective_positive_ratio": summary.get("effective_positive_ratio"),
                    }
                )
        if not rows:
            return missing(f"No finished run tagged `exp:{exp_id}` has a dataset summary yet.")
        return mo.ui.table(pd.DataFrame(rows), selection=None)

    return (composition_table,)


@app.cell
def _(composition_table):
    composition_table("S2d")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S3. Do adapters earn their cost?

    **The question.** Everything so far reads a frozen representation. If the
    run-up carries a signal the probe cannot reach, the limit might be the
    representation rather than the label or the selection rule. Low-rank
    adapters let the representation move, at a cost per configuration orders of
    magnitude higher.

    **Why it runs at one depth.** Sweeping depth inside a run works because the
    stored activations hold every layer. An adapted run has no stored
    activations, so it carries one probe at one depth and places its adapters
    up to that depth. The depth is the one S1 selected.

    **Why the control is its own run.** The comparison needs a frozen run at
    the same depth trained the same way, not one head of a many-headed run. A
    many-headed run selects its checkpoint on whichever depth leads and fits one
    class weight across all of them, so it differs from an adapted run in more
    than the regime.

    **What went wrong the first time, and what changed.** An earlier attempt
    was already worse than frozen at its first evaluation and got steadily
    worse, with validation loss climbing while ranking stayed high. That is a
    randomly initialised probe pushing gradients into the adapters before it
    knows what it is looking for: the ordering survives, the calibration does
    not, and a few confidently wrong negatives push the threshold up and
    collapse recall. The adapters now move an order of magnitude more slowly
    than the head. Freezing them outright for the first few hundred steps is the
    stronger version of the same fix and is not implemented; if the slower rate
    is not enough, that is the next thing to build.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("S3")
    return


@app.cell
def _(run_status):
    run_status("S3")
    return


@app.cell
def _(curve_figure, layer_choice):
    curve_figure("S3", layer_choice.value)
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("S3", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # S4. The held-out test, once

    **The question.** Everything up to here was chosen on validation data. This
    is the only number that says how the winner behaves on data nothing was
    tuned against.

    **How it runs.** No training. The winning recipe is scored on the two test
    splits using the thresholds already frozen on validation, applied
    unchanged. The reporting tool refuses to produce a test report for a scorer
    with no frozen thresholds, so the leak is structurally impossible rather
    than merely discouraged.

    **How it is read.** Held-out domains are reported per domain and never
    pooled. Beside the frozen-threshold numbers sits the threshold-free ranking
    for each domain, because the two together separate a calibration shift, in
    which the ordering still works and the threshold no longer fits, from a
    representation failure, in which the ordering itself does not transfer.
    Those call for different fixes and one number cannot tell them apart.

    Any per-domain cell backed by very few positive rollouts is marked
    anecdotal rather than quoted as a rate.

    **After this, nothing is tuned.** A second pass over the test splits with a
    different recipe would make them a validation set with extra steps.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("S4")
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("S4", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Notes for whoever picks this up next

    Facts that are expensive to rediscover, including for an AI assistant
    reading this notebook as context.

    **Never run training or scoring on the login node.** It is shared.
    Everything heavy goes through `sbatch`. Quick things (file listings,
    `git`, the unit tests, `evaluate_scores.py`) are fine directly.

    **Two Python environments.** `.venv/bin/python` at the repo root runs
    the tests and the cheap analysis scripts. Cluster jobs run inside a
    container described by `cluster/env.toml`; the image is built from
    `cluster/Dockerfile` via `cluster/build.sh`.

    **The test suite is the fastest way to check a change.**
    `.venv/bin/python -m pytest tests -q`, about 15 seconds.

    **Data layout.** The build root is
    `/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/degeneration-dataset-apertus-8b-instruct`.
    Activations live one file per rollout, shape `[33, tokens, 4096]` in
    fp16. **Slot 0 is the embedding**, so probe layer $L$ is cached slot
    $L + 1$. Getting this wrong silently trains on the neighbouring layer,
    which still works well enough to look plausible.

    **Ground truth.** The frontier is the judge's onset quote, located in
    the token stream and cached in
    `onset_labels/onset_quote_positions.parquet`; `onset_labels.parquet` is
    what training reads. 818 of 890 capped rollouts carry a resolved onset.
    The rest are excluded rather than treated as either class, and each
    records why. Those exclusions are concentrated in one domain, so
    per-domain populations are not proportional to the corpus.

    **Never read an onset out of any other column.** One function owns the
    definition; everything else goes through it.

    **Known traps.**

    - A trailing window with horizon 0 contains no positive token. The
      dataset refuses to build it rather than training on nothing.
    - `val/loss` carries a class weight fitted to each recipe's own
      training stream, so it is not comparable between recipes. Use
      `val/loss_unweighted`.
    - The token budget per step is derived from the *measured* tokens per
      example, not from the configured window size. In the adapted regime a
      selection rule masks the loss instead of shrinking the batch, so an
      example there is a whole rollout.
    - One held-out domain contributes a single positive rollout. It can
      never support a per-domain rate.
    - Runs before August 2026 were trained against a different frontier
      definition and their evaluation numbers are not comparable to
      anything here.

    **Finding runs.** Every run writes `run_info.json` with its axes, tags,
    group and status. The group is the run name minus the seed, so seed
    repeats aggregate. This notebook keys off the `exp:<id>` tag; the
    broader inventory of all runs, regardless of experiment, lives in
    `notebooks/inspect_runs.py`, which is for inspection rather than
    narrative.

    **Figures.** Everything drawn here is written to
    `notebooks/figures/diary/` as both PDF and PNG, so a figure can go
    straight into the paper.
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
        # One line per distinct configuration, whatever number of experiments
        # it belongs to, so nothing is submitted twice into one directory.
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
        Every command in this notebook, written to one script:

        ```
        {launcher_path}
        ```

        It is regenerated whenever this cell runs, so the register above stays
        the single source of truth. Submitting it wholesale queues every
        experiment that has no dependency on another's result.
        """
    )
    return


if __name__ == "__main__":
    app.run()
