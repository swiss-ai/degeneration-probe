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
    | `val/rollout_auc` | Ranking quality over rollouts, using each rollout's peak score | **The selection metric.** Rank-based, so unaffected by class weighting. 0.5 is chance, 1.0 is perfect separation |
    | `val/rollout_ap` | Average precision over the same ranking | More sensitive than AUC when positives are rare |
    | `val/loss` | The training objective, evaluated on validation | Weighted by a class weight fitted to *this recipe's* training stream. **Not comparable between recipes** |
    | `val/loss_unweighted` | The same loss with the weight removed | Measured identically for every recipe, so this is the one to compare across runs |
    | `val/prediction_std` | Spread of the probe's scores | **The collapse alarm.** Near zero means the probe outputs a constant, which converges nicely and distinguishes nothing. Guarded at 0.01 |
    | `val/target_mean` | Fraction of validation tokens that are positive | Fixed across recipes, because validation always uses the full split |
    | `val/prediction_mean` | Average score emitted | Far from `target_mean` means miscalibration, which the threshold step later absorbs |
    | `train/active_tokens` | Tokens actually reaching the loss, per forward pass | Times the accumulation, this is the realized token budget per step |
    | `train/pos_weight` | The class weight in use | Derived from the training stream's own balance |

    Two things worth knowing before comparing runs on these.

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
    # Settings are held as a mapping rather than a list of override strings, so
    # that two recipes are the same recipe exactly when their settings match. A
    # list would let one run spell out a default that another leaves implicit,
    # and the two would look different while training the identical probe.
    # Every run carries a head at every depth rather than picking one. Reading
    # the residual stream is what a run spends its time on, and that cost is
    # paid once whether one head or thirty-one hang off it, so depth stops
    # being a variable that has to be swept and becomes an axis every result
    # already has. Depth is then chosen when a result is read, not before it
    # is measured.
    PROBED_LAYERS = list(range(1, 32))
    # Scoring writes one file per token per depth, so it is done for a few
    # depths rather than all of them: the best found so far, a shallower
    # neighbour to show the plateau is broad, and a late one to show the fall.
    SCORED_LAYERS = [8, 12, 30]
    DEFAULT_LAYER = 12

    BASE = {
        "training.features.regime": "cached",
        "training.lora.enabled": "false",
        # Equal budget: the same tokens per optimizer step and the same number
        # of steps for every recipe, so the total tokens seen is held constant
        # and only the choice of tokens varies.
        "training.budget.tokens_per_step": 2048,
        # Long enough that every depth has stopped improving. The good depths
        # settle by step 400 and the late ones keep creeping to about 600, so
        # this leaves margin without paying for it twice.
        "training.runtime.max_steps": 800,
        "training.runtime.per_device_train_batch_size": 8,
        "training.validation.strategy": "steps",
        "training.validation.steps": 200,
        "training.checkpoint.strategy": "steps",
        "training.checkpoint.steps": 200,
        # Monitoring reads a fixed subsample rather than the whole split. The
        # full split costs more than the training it is watching once every
        # depth is scored on it.
        "training.validation.max_rollouts": 400,
        # A run writes no end-of-run evaluation. Scoring happens afterwards
        # from a checkpoint, one depth at a time, so evaluating every split at
        # every depth inside the run would be work nothing reads.
        "training.validation.final_splits": "[]",
        # Ranking saturates long before a probe is useful, so the checkpoint is
        # chosen on the operating point instead.
        "training.checkpoint.metric_for_best_model": "recall_at_budget",
        "training.runtime.seed": 42,
        # Every setting any experiment varies is named here even when it equals
        # the configured default, so that a recipe is described the same way
        # wherever it appears and identical recipes collapse to one run.
        "training.probe.layers": "[" + ",".join(str(n) for n in PROBED_LAYERS) + "]",
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
    SEEDS = [42, 43, 44]
    return BASE, DEFAULT_LAYER, LADDER_RUNGS, PROBED_LAYERS, SCORED_LAYERS, SEEDS


@app.cell
def _(BASE, LADDER_RUNGS, SEEDS):
    def recipe(**changes):
        return {**BASE, **changes}

    def _ladder_runs():
        runs = []
        for rung, strategy in enumerate(LADDER_RUNGS, start=1):
            for seed in SEEDS:
                runs.append(
                    {
                        "label": f"rung {rung}: {strategy} (seed {seed})",
                        "overrides": recipe(
                            **{
                                "training.selection.strategy": strategy,
                                "training.runtime.seed": seed,
                            }
                        ),
                    }
                )
        return runs

    def _window_runs():
        runs = []
        for strategy in ("random_window", "frontier_window"):
            for window in (64, 128, 256):
                runs.append(
                    {
                        "label": f"{strategy}, W={window}",
                        "overrides": recipe(
                            **{
                                "training.selection.strategy": strategy,
                                "training.selection.window_size": window,
                            }
                        ),
                    }
                )
        return runs

    def _layer_runs():
        # Depth is no longer swept. One run carries a head at every depth, so
        # the depth question is answered by reading one run along its layer
        # axis, and this recipe is the same one the ladder trains at rung four.
        # Planning it here tags that single run for both experiments rather
        # than training it twice.
        return [
            {
                "label": "every depth at once",
                "overrides": recipe(
                    **{"training.selection.strategy": "frontier_window"}
                ),
            }
        ]

    def _label_runs():
        # The window has to be wide enough to express the horizon, or the
        # comparison measures the window instead. A centered window spends half
        # its length after the frontier and so shows only W/2 tokens of run-up,
        # which means it needs W >= 2N; a trailing window is all run-up and
        # needs W >= N. Below that, two different horizons label every token in
        # the window positive and train on identical data, and the two runs
        # would be reported as a pair of points showing no difference.
        #
        # W is therefore fixed at 512 across the whole experiment, wide enough
        # for the largest horizon at either anchor, and the tokens per step are
        # raised to match so that a step still covers the same number of
        # windows as elsewhere.
        WIDE = {
            "training.selection.strategy": "frontier_window",
            "training.selection.window_size": 512,
            "training.budget.tokens_per_step": 8192,
        }
        runs = [
            {
                "label": f"centered, horizon {horizon}",
                "overrides": recipe(**WIDE, **{"training.label.horizon": horizon}),
            }
            for horizon in (0, 32, 128, 256)
        ]
        # A trailing window sits entirely before the frontier, so a horizon of
        # zero would leave no positive token in it at all. Only horizons that
        # reach back into the window are defined here.
        runs += [
            {
                "label": f"trailing, horizon {horizon}",
                "overrides": recipe(
                    **WIDE,
                    **{
                        "training.selection.anchor": "trailing",
                        "training.label.horizon": horizon,
                    },
                ),
            }
            for horizon in (128, 256, 512)
        ]
        return runs

    def _shape_runs():
        runs = [
            {
                "label": f"soft, {decay} over {length}",
                "overrides": recipe(
                    **{
                        "training.selection.strategy": "frontier_window",
                        "training.label.family": "frontier_soft",
                        "training.label.decay": decay,
                        "training.label.decay_length": length,
                        "training.loss.bce.use_pos_weight": "false",
                    }
                ),
            }
            for decay in ("exponential", "linear")
            for length in (128, 512)
        ]
        runs.append(
            {
                "label": "regression on repetition score",
                "overrides": recipe(
                    **{
                        "training.selection.strategy": "frontier_window",
                        "training.label.family": "token_signal",
                        "training.label.signal": "repetition_score",
                        "training.loss.name": "mse",
                    }
                ),
            }
        )
        return runs

    def _balance_runs():
        return [
            {
                "label": f"pos_weight {'on' if weight else 'off'}, positives {fraction}",
                "overrides": recipe(
                    **{
                        "training.selection.strategy": "frontier_window",
                        "training.loss.bce.use_pos_weight": str(weight).lower(),
                        "training.selection.positive_fraction": fraction,
                    }
                ),
            }
            for weight in (True, False)
            for fraction in (0.25, 0.5)
        ]

    def _budget_runs():
        # Long enough to run past the point where the metric stops improving,
        # and validated often enough to see where that happens. Every depth is
        # carried at once, so one run answers both how long to train and
        # whether the depth ranking is a property of the layer or only of when
        # training was stopped. The two are easy to confuse: a shallow depth
        # that has converged and a late depth that is still climbing look the
        # same at any single step.
        return [
            {
                "label": "every depth, 3000 steps",
                "overrides": recipe(
                    **{
                        "training.selection.strategy": "frontier_window",
                        "training.runtime.max_steps": 3000,
                        "training.validation.steps": 100,
                        "training.checkpoint.steps": 100,
                    }
                ),
                "sbatch": "--time=06:00:00",
            }
        ]

    budget_runs = _budget_runs()
    ladder_runs = _ladder_runs()
    window_runs = _window_runs()
    layer_runs = _layer_runs()
    label_runs = _label_runs()
    shape_runs = _shape_runs()
    balance_runs = _balance_runs()
    return (
        balance_runs,
        budget_runs,
        label_runs,
        ladder_runs,
        layer_runs,
        shape_runs,
        window_runs,
    )


@app.cell
def _(
    balance_runs,
    budget_runs,
    label_runs,
    ladder_runs,
    layer_runs,
    shape_runs,
    window_runs,
):
    EXPERIMENTS = [
        {
            "id": "E9",
            "title": "How long is long enough, and at what depth",
            "depends_on": [],
            "runs": budget_runs,
            "shell": [],
        },
        {
            "id": "E0",
            "title": "Baselines through the same protocol",
            "depends_on": [],
            "runs": [],
            # One job: score all three baselines on all three splits, then put
            # each through the evaluator with a persistence sweep.
            "shell": ["sbatch cluster/baselines.sbatch"],
        },
        {
            "id": "E1",
            "title": "How large should a window be?",
            "depends_on": [],
            "runs": window_runs,
            "shell": [],
        },
        {
            "id": "E2",
            "title": "The five-rung ladder",
            "depends_on": [],
            "runs": ladder_runs,
            "shell": [],
        },
        {
            "id": "E3",
            "title": "Which layer knows first?",
            "depends_on": [],
            "runs": layer_runs,
            "shell": [],
        },
        {
            "id": "E4",
            "title": "Buying lead time with the label",
            "depends_on": [],
            "runs": label_runs,
            "shell": [],
        },
        {
            "id": "E5",
            "title": "Soft labels, and regression instead of detection",
            "depends_on": [],
            "runs": shape_runs,
            "shell": [],
        },
        {
            "id": "E6",
            "title": "Class balance and calibration",
            "depends_on": [],
            "runs": balance_runs,
            "shell": [],
        },
        {
            "id": "E7",
            "title": "Do adapters earn their cost?",
            "depends_on": ["E1", "E2", "E3", "E4"],
            "runs": [],
            "shell": [],
        },
        {
            "id": "E8",
            "title": "The held-out test, once",
            "depends_on": ["E7"],
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
                    "best": training.get("best_metric"),
                    "selected_on": training.get("metric_for_best_model"),
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

    return (
        depth_table,
        load_curves,
        load_views,
        protocol_table,
        run_dirs_for,
        scored_layers,
    )


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

    Every recipe is given the same number of tokens per optimizer step and
    the same number of steps, so the total gradient is held constant and
    only the *choice* of tokens varies. Holding the total constant has a
    consequence worth being explicit about: the rules produce very
    different amounts of data, so a rule with a large pool never finishes a
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
    3. **Training reads that order in sequence** and stops at the step
       limit, taking a prefix of it.

    Because the shuffle happens in stage 2 and the cut in stage 3, **a
    prefix is a uniform random sample**, not the first rollouts in file
    order. A rule that gets through a tenth of its pool still draws that
    tenth from everywhere, across every domain and every position within a
    rollout.

    The table below reports what each rule actually receives. The column
    to read alongside the unique-window count is the number of *rollouts*
    touched: a rule can use a small share of its windows while still
    seeing nearly every degeneration episode, which is a very different
    situation from having seen only a small share of the episodes.
    """)
    return


@app.cell
def _(BASE, DEFAULT_LAYER, REPO, missing, mo, pd):
    def budget_coverage():
        import numpy as np
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from degeneration_probe.config import (
            ExperimentConfig,
            LabelConfig,
            SelectionConfig,
        )
        from degeneration_probe.data.dataset import load_degeneration_records
        from degeneration_probe.data.windowed_dataset import WindowedActivationDataset
        from degeneration_probe.training.arguments import resolve_token_budget

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

        batch = int(BASE["training.runtime.per_device_train_batch_size"])
        steps = int(BASE["training.runtime.max_steps"])
        tokens_per_step = int(BASE["training.budget.tokens_per_step"])
        defaults = OmegaConf.to_container(composed.training.selection, resolve=True)
        defaults.update(
            window_size=BASE["training.selection.window_size"],
            anchor=BASE["training.selection.anchor"],
            positive_fraction=BASE["training.selection.positive_fraction"],
        )
        positive_rollouts = sum(record.is_positive for record in records)

        rows = []
        for strategy in (
            "all_tokens",
            "rollout_balanced",
            "random_window",
            "frontier_window",
        ):
            dataset = WindowedActivationDataset(
                records,
                build_root=experiment.dataset.build_root,
                # Which depth is read does not change which windows the rule
                # builds, and this table is about the windows.
                probe_layer=DEFAULT_LAYER,
                selection=SelectionConfig(**{**defaults, "strategy": strategy}),
                batch_size=batch,
                seed=BASE["training.runtime.seed"],
            )
            resolved = resolve_token_budget(
                tokens_per_step,
                valid_tokens=dataset.summary()["valid_tokens"],
                examples=len(dataset),
                per_device_batch_size=batch,
            )
            consumed = steps * resolved["gradient_accumulation_steps"] * batch
            order = np.array(dataset.order)
            is_positive = np.array(
                [
                    dataset.records[dataset.windows[index].record_index].is_positive
                    for index in order
                ]
            )
            seen = order[:consumed]
            seen_positive = seen[is_positive[:consumed]]
            rows.append(
                {
                    "rule": strategy,
                    "windows built": len(dataset.windows),
                    "slots in one pass": len(order),
                    "slots consumed": min(consumed, len(order)),
                    "passes": round(consumed / max(1, len(order)), 2),
                    "positive windows used": len(set(seen_positive.tolist())),
                    "positive windows available": int(is_positive.sum()),
                    "positive rollouts touched": len(
                        {dataset.windows[int(i)].record_index for i in seen_positive}
                    ),
                    "of positive rollouts": positive_rollouts,
                }
            )
        return mo.ui.table(pd.DataFrame(rows), selection=None)

    budget_coverage()
    return


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

    The budget is only fair if it is not also starving anyone. The check is
    the training curve: a run whose selection metric is still climbing at
    the step limit was cut short rather than fairly constrained, and the
    budget needs raising for every recipe together.
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
    # is chosen here, when a result is read. Training curves exist at every
    # depth; protocol tables exist only where a run has been scored, which is
    # a few depths per run because scoring writes a score for every token.
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
    # E9. How long is long enough, and at what depth

    **The question.** Two, sharing one sweep, because neither answer is worth
    much without the other.

    Every recipe is trained on the same token budget so that the choice of
    tokens is the only thing separating them. That is only a fair constraint if
    it is not also a cut-off: a recipe still improving when the budget runs out
    was stopped, not constrained, and comparing it to another is comparing two
    unfinished runs. This sweep trains far past the point of interest and
    validates often enough to see where each curve flattens.

    The same runs vary the layer. Depth and training length are entangled,
    because a layer that looks weak may simply converge more slowly, and a
    short budget would report that as a property of the layer. Running several
    depths at a length where all of them have finished separates the two.

    **What to expect, and why.** The metric should rise quickly and then
    flatten; where it flattens is the budget every other experiment should use,
    taken across layers rather than from the fastest one. On depth, the
    expectation is a plateau over the middle layers with a falloff at both
    ends: early layers should not yet represent anything as abstract as "I am
    repeating myself", and the last layers are specialised for choosing the
    next token, which is a narrower job than describing the state of the
    generation.

    Read this before anything else. A comparison drawn from runs that were
    still improving describes the budget, not the recipes.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("E9")
    return


@app.cell
def _(run_status):
    run_status("E9")
    return


@app.cell
def _(FIGURES, load_curves, missing, plt):
    def budget_figure():
        frame = load_curves("E9")
        if frame.empty or "val/rollout_auc" not in frame.columns:
            return missing(
                "No finished run tagged `exp:E9` has written a validation curve yet."
            )
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        # One curve per depth. A single run carries all of them, so these are
        # the same optimizer steps over the same tokens throughout and the
        # curves differ only in where the probe reads.
        for label, group in frame.groupby("layer"):
            points = group[["step", "val/rollout_auc"]].dropna().sort_values("step")
            if not len(points):
                continue
            axes[0].plot(points["step"], points["val/rollout_auc"], marker="", label=label)
            # The same curve as distance from its own best, which is where a
            # plateau is legible: flat here means more steps buy nothing.
            best = points["val/rollout_auc"].cummax()
            axes[1].plot(points["step"], best.iloc[-1] - points["val/rollout_auc"], label=label)
        axes[0].set_ylabel("rollout AUC on validation")
        axes[0].set_title("does it flatten?")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("gap to the run's final best")
        axes[1].set_title("how far from converged")
        for axis in axes:
            axis.set_xlabel("step")
            axis.grid(alpha=0.3)
        axes[0].legend(fontsize=8, title="layer")
        fig.tight_layout()
        path = FIGURES / "E9_budget"
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        return fig

    budget_figure()
    return


@app.cell
def _(load_curves, missing, mo, pd):
    def plateau_table():
        frame = load_curves("E9")
        if frame.empty or "val/rollout_auc" not in frame.columns:
            return missing("No validation curve for any run tagged `exp:E9` yet.")
        rows = []
        for layer, group in frame.groupby("layer"):
            points = group[["step", "val/rollout_auc"]].dropna().sort_values("step")
            if not len(points):
                continue
            best = float(points["val/rollout_auc"].max())
            peak_step = int(points.loc[points["val/rollout_auc"].idxmax(), "step"])
            last_step = int(points["step"].iloc[-1])
            # The first step reaching within a small margin of the best is the
            # honest reading of "long enough": improvements past it are smaller
            # than the seed-to-seed spread anything will be compared against.
            within = points[points["val/rollout_auc"] >= best - 0.001]
            rows.append(
                {
                    "layer": int(layer),
                    "best AUC": round(best, 5),
                    "step of best": peak_step,
                    "first within 0.001": int(within["step"].iloc[0]) if len(within) else None,
                    "last step": last_step,
                    "still improving at the end": peak_step >= last_step - 100,
                }
            )
        return mo.ui.table(pd.DataFrame(rows).sort_values("layer"), selection=None)

    plateau_table()
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("E9", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # E0. Baselines through the same protocol

    **The question.** How well do simple, model-free signals already do?

    Three of them: the sliding repetition score, per-token entropy, and a
    longest-repeated-substring match. Each is turned into a per-token score
    and pushed through the identical evaluator, so the comparison is not
    rhetorical.

    **What to expect, and why.** Repetition scoring should be a genuinely
    strong detector, because by the time a rollout is looping, repetition
    is exactly what is there to see. Its weakness should be timing rather
    than detection: it needs the loop to exist before it can measure it, so
    it should fire late. Entropy should be weaker and noisier.

    This experiment also sets the persistence window $m$ for everything
    that follows. The command sweeps $m$ over several values and reports
    what each buys, which is why View D matters here: if a scorer's false
    alarms are one token long, a larger $m$ removes them almost for free;
    if they run for tens of tokens, no $m$ will help and the scorer is
    confidently wrong rather than jittery.

    Nothing depends on training, so this can run immediately.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("E0")
    return


@app.cell
def _(Path, missing, mo, pd):
    def baseline_table(split):
        rows = []
        for name in ("repetition", "entropy", "lrs"):
            path = Path("outputs/baselines") / name / "evaluation" / split / "view_a_detection.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "baseline", name)
            rows.append(frame)
        if not rows:
            return missing(
                f"No baseline has protocol output on `{split}` yet. "
                "Run the two commands above, in order."
            )
        return mo.ui.table(pd.concat(rows, ignore_index=True), selection=None)

    return (baseline_table,)


@app.cell
def _(baseline_table, split_choice):
    baseline_table(split_choice.value)
    return


@app.cell
def _(Path, missing, mo, pd):
    def persistence_comparison():
        frames = []
        for name in ("repetition", "entropy", "lrs"):
            path = Path("outputs/baselines") / name / "evaluation" / "persistence_comparison.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "baseline", name)
            frames.append(frame)
        if not frames:
            return missing(
                "No persistence sweep yet. It comes from `--compare-persistence` "
                "in the commands above and is written to "
                "`evaluation/persistence_comparison.csv`."
            )
        return mo.ui.table(pd.concat(frames, ignore_index=True), selection=None)

    return (persistence_comparison,)


@app.cell
def _(persistence_comparison):
    persistence_comparison()
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # E1. How large should a window be?

    **The question.** A training example is a contiguous strip of $W$
    tokens. Does $W$ matter, and which way?

    Six runs: $W \in \{64, 128, 256\}$ crossed with two rules, one that
    places the window at random and one that anchors it on the frontier.

    **What to expect, and why.** Repetition is a pattern *across* tokens,
    so a window too short to contain a full repeat cannot show the probe
    what a repeat looks like. That argues for larger $W$. Against it, the
    token budget per step is fixed, so doubling $W$ halves the number of
    independent examples per step, and windows drawn from the same rollout
    are correlated. Somewhere between those two effects there should be a
    best value, and it may differ between the two rules: an anchored window
    already contains the interesting region, so it may need less room than
    one placed at random.

    Whatever wins here is used for the ladder. The ladder is run at
    $W = 128$ regardless, so the two experiments can proceed in parallel;
    if this one says otherwise, the ladder is rerun at the better value.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("E1")
    return


@app.cell
def _(run_status):
    run_status("E1")
    return


@app.cell
def _(curve_figure, layer_choice):
    curve_figure("E1", layer_choice.value)
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("E1", split_choice.value, layer_choice.value)
    return


@app.cell
def _(layer_choice, operating_figure, split_choice):
    operating_figure("E1", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # E2. The five-rung ladder

    **The question.** Which tokens should the probe be trained on? This is
    the central experiment.

    Five rules, each changing exactly one decision relative to the one
    before it, every rung at three seeds and on an identical token budget:

    1. **all_tokens** — every token of every rollout
    2. **rollout_balanced** — a fixed budget per rollout, drawn at random
    3. **random_window** — the same budget, now contiguous
    4. **frontier_window** — the same window, anchored on the frontier
    5. **+ hard negatives** — negative windows biased toward repetitive spans

    Because adjacent rungs differ by one decision, the difference between
    them is attributable to that decision. Differences are read against the
    seed-to-seed spread of the two groups added in quadrature; anything
    smaller than that spread is not a result.

    **What to expect, and why.** The largest single step should be at rung
    4. Up to that point a probe can score well by learning "this text is
    long and repetitive", which is a property of the whole rollout rather
    than of the moment degeneration begins. A window straddling the
    frontier contains both sides of that moment and removes the shortcut.

    Hard negatives at rung 5 should show up in the false-alarm rate and in
    View D rather than in AUC: they are aimed at the specific error of
    mistaking ordinary repetitive text for a loop.

    The risk worth naming in advance: rung 1 may already be close to
    ceiling on detection, in which case the ladder has to be read on
    precision, lead time and false alarms, and the AUC column will say
    nothing.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("E2")
    return


@app.cell
def _(run_status):
    run_status("E2")
    return


@app.cell
def _(curve_figure, layer_choice):
    curve_figure("E2", layer_choice.value)
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("E2", split_choice.value, layer_choice.value)
    return


@app.cell
def _(layer_choice, operating_figure, split_choice):
    operating_figure("E2", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    Pooled over seeds, and read as adjacent-rung differences. The
    `_beats_noise` flag is true only when a difference exceeds the spread
    it came from.

    ```bash
    .venv/bin/python scripts/compare_runs.py --split val --ladder     <group of rung 1> <group of rung 2> <group of rung 3>     <group of rung 4> <group of rung 5>
    ```

    The group names are in the status table above.
    """)
    return


@app.cell
def _(layer_choice, missing, mo, pd, run_dirs_for, split_choice):
    def pooled_ladder(split, layer):
        try:
            from degeneration_probe.analysis.run_comparison import (
                collect_results,
                collect_runs,
                pool_seeds,
            )
        except ImportError:
            return missing("`degeneration_probe` is not importable from this kernel.")
        if not run_dirs_for("E2"):
            return missing("No finished run tagged `exp:E2` yet.")
        runs = collect_runs("outputs")
        if runs.empty:
            return missing("No runs under `outputs/`.")
        results = collect_results(runs[runs["status"] == "finished"], split, layer)
        if results.empty:
            return missing(
                f"No protocol output on `{split}` at layer {layer} for any run."
            )
        pooled = pool_seeds(results)
        return mo.ui.table(pd.DataFrame(pooled).round(4), selection=None)

    pooled_ladder(split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # E3. Which layer knows first?

    **The question.** The probe reads one layer. Which one, and how much
    does it matter?

    Nine runs across the depth of a 32-layer model. This is the cheapest
    high-information experiment available, because activations for every
    layer of every rollout are already on disk: the sweep costs training
    time only.

    **What to expect, and why.** A broad plateau over the later-middle
    layers, falling off at both ends. Very early layers should not yet
    represent anything as abstract as "I am repeating myself"; the very
    last layer is specialised for predicting the next token, which is a
    narrower job than describing the state of the generation.

    The interesting outcome is the one that would be useful rather than the
    one that is expected: if a much earlier layer works nearly as well as a
    late one, a deployed probe could run on a truncated forward pass and
    cost a fraction of a full one.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("E3")
    return


@app.cell
def _(run_status):
    run_status("E3")
    return


@app.cell
def _(FIGURES, missing, plt, tagged_with):
    def layer_figure():
        frame = tagged_with("E3")
        if frame.empty or "status" not in frame.columns:
            return missing("No run tagged `exp:E3` found under `outputs/`.")
        frame = frame[frame["status"] == "finished"].dropna(subset=["layer", "best"])
        if frame.empty:
            return missing("No finished run tagged `exp:E3` has a best metric yet.")
        frame = frame.sort_values("layer")
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.plot(frame["layer"], frame["best"], marker="o")
        axis.set_xlabel("probe layer")
        axis.set_ylabel(frame["selected_on"].iloc[0] or "best metric")
        axis.set_title("selection metric by depth")
        axis.grid(alpha=0.3)
        fig.tight_layout()
        path = FIGURES / "E3_layers"
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        return fig

    layer_figure()
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("E3", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    # E4. Buying lead time with the label

    **The question.** An alarm is only useful if it arrives in time. Can
    the label be used to buy earliness, and what does it cost?

    Two things vary together, because they interact. The **horizon** $N$
    marks the $N$ tokens before the frontier as positive too, explicitly
    teaching the probe to fire early. The **anchor** decides whether the
    training window straddles the frontier or sits entirely before it.

    A trailing window with a horizon of zero contains no positive token at
    all, so that combination is refused at dataset construction rather than
    trained silently. Only horizons that reach back into the window appear
    here.

    **What to expect, and why.** A clean trade. Larger $N$ should move the
    median alarm earlier and cost precision, because tokens that are not
    yet degenerate are being labelled as if they were. The trailing anchor
    asks the sharper question: can the run-up alone predict the loop, with
    the probe never having seen the loop itself during training? If it can,
    that is the strongest form of the result, since it cannot be explained
    by the probe recognising repetition it has already been shown.

    Read this one against View C rather than View A, and keep in mind that
    the frontier itself carries positional uncertainty, so differences of a
    few tokens mean nothing.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("E4")
    return


@app.cell
def _(run_status):
    run_status("E4")
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("E4", split_choice.value, layer_choice.value)
    return


@app.cell
def _(layer_choice, operating_figure, split_choice):
    operating_figure("E4", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # E5. Soft labels, and regression instead of detection

    **The question.** Does degeneration have to be framed as a moment at
    all?

    Two alternatives to the hard step. **Soft labels** ramp the target up
    towards the frontier instead of switching at it, over two shapes and
    two lengths. **Regression** abandons the frontier entirely and predicts
    a continuous per-token repetition score.

    **What to expect, and why.** Soft labels should land between the hard
    horizons of E4, with a smoother score trajectory. That smoothness
    should help persistence specifically: an alarm built from a gradually
    rising score is less likely to flicker than one built from a step.

    Regression is a different task wearing the same clothes. It predicts
    *how repetitive* rather than *whether degenerate*, so it should do
    relatively better on coverage (View B) and relatively worse on rollout
    detection (View C and A). It is included as a check that the frontier
    framing earns its place, not because it is expected to win.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("E5")
    return


@app.cell
def _(run_status):
    run_status("E5")
    return


@app.cell
def _(layer_choice, protocol_table, split_choice):
    protocol_table("E5", split_choice.value, layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # E6. Class balance and calibration

    **The question.** Positives are rare. Two mechanisms correct for that,
    and they can be applied together: over-sampling positives when
    composing batches, and weighting the positive class in the loss. Does
    either help, and does applying both correct twice?

    Four runs: the class weight on and off, crossed with two positive
    fractions.

    **What to expect, and why.** Very little movement in AUC, which is
    rank-based and largely indifferent to how the classes are weighted.
    Substantial movement in where the raw scores sit, which the threshold
    step then absorbs, since thresholds are chosen by budget rather than
    fixed at 0.5.

    The quantity to watch is the effective ratio recorded beside each run:
    it lands on one when the weight matches the sampled population, and
    departs from one when the imbalance has been corrected twice or not at
    all. This is also the experiment that shows why the unweighted
    validation loss exists, since the weighted one moves here for reasons
    that have nothing to do with the probe.
    """)
    return


@app.cell
def _(show_commands):
    show_commands("E6")
    return


@app.cell
def _(run_status):
    run_status("E6")
    return


@app.cell
def _(Path, json, missing, mo, pd, run_dirs_for):
    def composition_table(exp_id):
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

    composition_table("E6")
    return


@app.cell
def _(curve_figure, layer_choice):
    curve_figure("E6", layer_choice.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # E7. Do adapters earn their cost?

    **The question.** Every experiment so far trains a head on frozen,
    cached activations. Letting the model itself adapt costs roughly ten
    times as much. Is it worth it?

    The winning recipe from E1 to E4, run both ways at three seeds.

    **What to expect, and why.** Adapters should buy something, since they
    can reshape the representation towards the task rather than merely
    reading it. The question is whether the gain clears the seed-to-seed
    noise. If it does not, that is the most useful negative result in the
    whole plan: every future sweep stays cheap, and the claim becomes the
    stronger one that the signal is already present in the unmodified
    model rather than manufactured by fine-tuning.

    **Depends on E1, E2, E3 and E4**, since "the winning recipe" is their
    output. Commands appear once those are read.
    """)
    return


@app.cell
def _(run_status):
    run_status("E7")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # E8. The held-out test, once

    **The question.** Does any of this survive contact with data that
    played no part in choosing it?

    The single chosen recipe, scored on the in-domain test split and on
    held-out domains, with thresholds frozen from validation and the
    baselines carried through the identical protocol.

    **What to expect, and why.** A drop from validation to held-out
    domains, because held-out domains are the real test of whether the
    probe learned degeneration or learned one domain's flavour of it.
    Per-domain results must respect the minimum-positive guard: at least
    one domain contributes almost no positive rollouts and cannot support
    a rate at all.

    **This runs once.** Every choice must already be frozen before it does,
    which is what the whole validation-only threshold discipline exists to
    protect. Re-running it after seeing the result would quietly turn the
    test split into a second validation set.

    **Depends on E7.**
    """)
    return


@app.cell
def _(run_status):
    run_status("E8")
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
