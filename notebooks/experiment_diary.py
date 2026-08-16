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

    It comes in three parts. **Part 1** sets out the problem: the corpus, what a
    probe is, and how one is measured. **Part 2** is the experiment register,
    one stage at a time, each stage ending in a decision the next one needs.
    **Part 3** asks whether a head trained on one model still works on a model
    it has never seen.

    Every control in this notebook belongs to the one figure or table it sits
    above. Nothing is steered from a single setting at the top, so two plots
    can be read at different depths or different budgets at the same time
    without one of them silently changing under the other.
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
    quote in the token stream.

    **A probe.** A linear head reading the residual stream at one layer,
    producing a score between 0 and 1 for every token. About twelve thousand
    parameters.

    **A rule for which tokens to train on.** A generation has thousands of tokens
    and almost all of them say nothing. We need to decide how to choose the tokens
    to train on, here five rules to form a ladder: from "use
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
    # Part 1. What degeneration is, and how it is measured

    ## 1a. The corpus
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

        # Read down the rows rather than across a row of quantile names: the
        # question this answers is "how far into an answer does the loop
        # start", and each row is one way of finishing that sentence.
        onsets = labels.loc[labels["is_positive"], "onset_position"]
        shape = pd.DataFrame(
            [
                {
                    "how far into the answer the loop starts": description,
                    "tokens": int(onsets.quantile(quantile)),
                }
                for description, quantile in [
                    ("earliest of any degenerate answer", 0.0),
                    ("a quarter of them start looping before", 0.25),
                    ("half of them start looping before", 0.50),
                    ("three quarters start looping before", 0.75),
                    ("latest of any degenerate answer", 1.0),
                ]
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
                Where the loop starts is the single most important fact about
                this dataset, and most design decisions follow from it. Sort
                every degenerate answer by the token at which its loop begins,
                and the table below reads off five points along that order.
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
    mo.md(r"""
    ---
    ## 1b. What a probe is, and how it is trained

    A probe is a normalization followed by one linear map. For the residual
    state $h_t^{(\ell)} \in \mathbb{R}^{4096}$ that layer $\ell$ produces at
    token $t$,

    $$
    z_t = w^\top \mathrm{LN}\!\left(h_t^{(\ell)}\right) + b,
    \qquad
    p_t = \sigma(z_t) = \frac{1}{1 + e^{-z_t}}
    $$

    where $\mathrm{LN}$ is a learned LayerNorm, $\sigma$ is the logistic
    function, and $p_t \in (0, 1)$ is the score the evaluator reads. The
    trained parameters are the LayerNorm's gain and shift, $2 \times 4096$,
    plus the map's weight and bias, $4096 + 1$: **12,289 in total**. Nothing
    else moves. The score at a token is a function of that token's state alone,
    which is what makes the probe deployable one token at a time, and it is
    built only from tokens up to $t$, so a probe never reads text it has not
    yet generated.

    Activations are cached to disk once, with the language model frozen, so
    training reads them back and never runs the model. That is what makes wide
    sweeps affordable.

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

    ### What goes into a batch

    **The training generations are not the corpus.** Negatives outnumber positives
    by more than a hundred to one. Left alone, that ratio decides the loss
    before the representation does, so the training split is first cut to
    **four negative generations per positive**, drawn **stratified by domain** so
    the cut keeps each domain's share of the negatives rather than whichever
    domain happens to be largest.

    Batches are then **composed rather than shuffled**. Each batch is a fixed
    fraction of positive windows, `positive_fraction`, a quarter by default,
    with the remaining three quarters negative and again drawn in proportion to
    each domain's share. Shuffling would leave the fraction to chance and let
    it drift between steps, which is exactly the quantity S2d exists to test.
    The class weight in the loss is a second correction of the same imbalance,
    which is why the two are crossed rather than assumed to be independent.

    Validation and test are never subsampled this way. They keep the corpus's
    own proportions, because a false-alarm rate measured on a re-balanced
    population would not be the rate a deployment sees.

    **Why the run tables count epochs oddly.** A selection rule cuts each
    answer into windows, and the positive windows are the finite pool the
    sampler draws a quarter of every batch from. Once it has used all of them
    it starts again from the beginning. An epoch here counts those laps: one
    epoch is one lap through the pool of positive windows, not one pass over
    the corpus.

    How long a lap is depends entirely on the rule. A rule that takes a single
    window per degenerate answer has a pool of a few hundred, so a 2000-step
    run goes round it many times and sees the same windows again and again. A
    rule that tiles every token of every answer has a pool of hundreds of
    thousands, and the same run never reaches the end of it once. So an epoch
    count below 1 means the run never ran out of fresh positive windows, and a
    count of 30 means it saw each of them about thirty times.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1c. How a probe is measured

    All reported numbers come from the black-box evaluator. It takes one score
    per token and knows nothing about what produced them. Probes and model-free
    heuristics therefore go through an identical judgement.

    **Everything below reads the same fixed population**: 3,634 validation
    answers, 108 of them degenerate. That is the validation split entire, bar
    six answers whose activations are not on disk. It is written down in
    `configs/dataset/validation_rollouts_<dataset>.csv` rather than resampled
    at runtime, and both readings of a checkpoint use that file: a full scoring
    pass and a replay over saved checkpoints see the identical 3,634 answers,
    so their numbers can be put beside each other.

    **How to choose a threshold for the scores** Scores from probes are in [0, 1]
    we need to choose a treshold for computing the binary label. Instead of choosing
    directly the threshold we fix the share of healthy generations allowed to raise a false alarm, at 1%, 5% and 10%, and read off the threshold that spends exactly that. Thresholds are
    chosen on validation and reused unchanged on test.

    **An alarm needs persistence.** A single token above threshold is usually
    noise, so an alarm requires $m$ consecutive tokens above it. The first alarm
    is

    $$a_r(\tau, m) = \min\{t : p_r(t') \ge \tau \ \text{for all} \ t' \in [t, t+m)\}$$

    Everything on disk today uses $m = 1$. (i.e. essentially not used, but could be used for future experiments)

    ### The four views

    #### View A. Detection: does it fire on the right answers?

    *Why.* This is the obvious question and the one that settles least.
    Roughly four fifths of a degenerate answer is already loop by the time it
    ends, and an answer's score is the highest any of its tokens reaches, so
    almost anything catches almost every degenerate answer. The view exists to
    confirm a scorer is not broken, not to rank scorers, and a stage that
    separated its recipes on recall alone would be separating them on noise.

    *Metrics* (rollout-level): **recall**, the share of true degenerate answers that raise an alarm from the probe,
    and **precision**, the share of alarms that were raised on a true degenerate
    answer.

    #### View B. Coverage: what share of tokens does it flag, and where?

    *Why.* Because a token deep inside the loop and a token in the run-up are
    not the same measurement, and pooling them lets the easy tokens decide the
    number. Splitting them is what turns "it can tell a broken answer from a
    good one" into the question this project is actually about.

    - **In-pattern coverage**, `in_pattern_recall`: of the tokens at or after
      the frontier, the share flagged. Whether the probe sees the obvious. The
      easy half, and a floor to clear rather than a result.
    - **Warning coverage**, `warning_recall_128` and `warning_recall_256`: of
      the tokens in a short band immediately *before* the frontier, the share
      flagged. Whether the probe sees the approach. Reported at two widths
      because how much warning is worth having is a judgement, and one column
      would hide it.
    - **Healthy token rate**, `token_false_positive_rate`: of every token in a
      **healthy answer**, the share flagged. Healthy answers only. The tokens
      before the frontier of a degenerate answer are not counted here, because
      on those tokens an alarm is early rather than wrong, and that is what
      warning coverage is for. Coverage without this rate is unreadable, since
      flagging every token maximises both of the numbers above.

    Warning coverage is the number to rank on. It is an average over tens of
    thousands of tokens, so it moves smoothly, and it is defined over every
    degenerate answer, so one that is never flagged lowers it instead of
    disappearing from it.

    One edge case, since it comes up: about 3% of degenerate answers start
    looping at the very first token. Warning coverage asks what share of the
    tokens *before* the loop were flagged, and those answers have no tokens
    before the loop. There is nothing to flag and nothing to miss, so they add
    zero to the flagged count and zero to the total, and the rate is simply
    taken over the other answers. They still count in every other view.

    #### View C. Lead time: how early is the alarm?

    *Why.* Lead time is the number a reader wants, and it is honest as a
    result. It is a bad thing to select on, because it is a median over only
    the answers that happened to fire: dropping the hard, late-firing answers
    improves it. So it is reported, and nothing is chosen by it.

    *Metric.* **median_offset**, the signed distance in tokens from the alarm
    to the true frontier, where **negative is early**. Signed rather than
    absolute because the sign is the whole claim: firing 200 tokens late is a
    different failure from firing 200 tokens early, and an unsigned average
    would call them equal.

    **never_fired_positives** is reported beside it rather than folded into it.
    An answer that never fires has no offset, and any single number that
    absorbed it would have to invent one. Counting those answers separately
    keeps the median an honest statement about the answers it describes, and
    makes the population it was taken over visible.

    #### View D. Persistence: having fired, does it stay convinced?

    *Why.* To separate a scorer that saw something from one that twitched. The
    view is read in **opposite directions for the two populations**, which is
    the reason it is a view of its own: on degenerate answers a long first
    alarm is good, while on healthy answers every alarm is wrong and its length
    says which kind of wrong. A short false alarm is jitter, which a larger $m$
    would remove; a long one is a confident mistake, which no $m$ can.

    *Metric.* **median_first_run_length**, the number of consecutive tokens the
    first alarm holds, reported once per population.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### Which checkpoint gets reported

    A run trains for 2000 steps and keeps a checkpoint every 50. Which of those
    forty checkpoints a run is judged on is decided by the rule below, applied
    to each depth independently, since the heads share no parameters and plateau
    at very different steps. It replaces an earlier rule that watched
    rollout-level recall during training, which saturated and could not separate
    one checkpoint from another.

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

    ### What the rule is for

    **Only the checkpoint the rule picks is ever scored in full.** A *full
    scoring pass* means writing one score for every token of all 3,634
    validation answers and putting them through all four views above. It takes
    a few minutes for one depth at one checkpoint. A run has 31 depths and 40
    checkpoints, so doing it everywhere would cost days per run and throw away
    all but one row of the result. The rule exists so that cost is spent once,
    at the checkpoint worth spending it on.

    **The rule is one function, usable at either moment.** It reads a sequence
    of cheap per-checkpoint measurements and says where a depth stopped
    improving and which checkpoint was its best. That sequence can arrive as
    training produces it, in which case the rule is an early-stopping
    criterion, or it can be read back afterwards from saved checkpoints, in
    which case the rule picks which one to report. The runs in this notebook
    are the second case, because they finished before the rule existed. Since
    the rule is the same code either way, what it decides for them is what it
    would have decided while they ran.
    """)
    return


@app.cell
def _(Path, all_runs, missing, mo, pd):
    from degeneration_probe.evaluation.head_selection import (
        StoppingRule,
        apply_rule_to_run,
        run_length,
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
                    f"{rule.tolerance}, patience {rule.patience}. One row per "
                    "replayed run: **never selectable** is how many of its 31 "
                    "depths never cleared the in-loop-coverage floor at all, "
                    "**earliest/median stop** is when its depths' patience ran out, "
                    "**best depth** is the one the run would actually be reported "
                    "at. A run ends when its **slowest** depth stops, so that "
                    "column is what a future run's walltime has to cover."
                ),
                mo.ui.table(pd.DataFrame(rows), selection=None),
            ]
        )

    return (stopping_outcomes,)


@app.cell
def _(stopping_outcomes):
    stopping_outcomes()
    return


@app.cell
def _(Path, all_runs, pd):
    # The quantity selection runs on, so the quantity whose shape decides whether
    # a longer run would have changed the answer.
    SELECTION_METRIC = "warning_recall_256"

    def replay_history():
        """Every replayed checkpoint of every run, carrying the run's settings."""
        if all_runs.empty:
            return pd.DataFrame()
        frames = []
        for row in all_runs.itertuples():
            path = Path(row.run_dir) / "checkpoint_replay.parquet"
            if not path.is_file():
                continue
            frame = pd.read_parquet(path)
            frame["run"] = Path(row.run_dir).parent.name
            frame["rule"] = row.selection
            frame["window"] = row.window
            frame["seed"] = row.seed
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    replay_frame = replay_history()
    return SELECTION_METRIC, replay_frame


@app.cell
def _(Path, all_runs, mo, pd, replay_frame):
    # Ninety-odd replayed runs is too many for one dropdown, so a run is chosen
    # in stages instead: the recipe, then the window size, then the seed. Window
    # and seed are not enough on their own to name a run, since several stages
    # reuse the same rule at the same width with a different label or horizon,
    # so the recipe carries every axis except those two.
    _RECIPE_AXES = [
        "selection",
        "anchor",
        "label",
        "horizon",
        "decay",
        "decay_length",
        "signal",
        "loss",
        "pos_weight",
        "positive_fraction",
        "features",
    ]

    def _axis_value(row, axis):
        value = getattr(row, axis, None)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return value

    def _build_replay_index():
        """One row per replayed run: how to name it, and how to find it again."""
        if replay_frame.empty or all_runs.empty:
            return pd.DataFrame(columns=["run", "recipe", "window", "seed"])
        replayed = set(replay_frame["run"].unique())
        rows = []
        for row in all_runs.itertuples():
            name = Path(row.run_dir).parent.name
            if name not in replayed:
                continue
            values = {axis: _axis_value(row, axis) for axis in _RECIPE_AXES}
            rows.append(
                {
                    "run": name,
                    "seed": _axis_value(row, "seed"),
                    "window": _axis_value(row, "window"),
                    **values,
                }
            )
        frame = pd.DataFrame(rows).drop_duplicates(subset="run")
        # Only the axes that actually differ between these runs go into a name.
        # Spelling out the ones every run shares would push the part that
        # distinguishes them off the end of the dropdown.
        varying = [
            axis
            for axis in _RECIPE_AXES
            if frame[axis].astype(str).nunique(dropna=False) > 1
        ]
        # Some settings are recorded on every run but only mean anything under
        # one label family: a hard-label run carries a decay length it never
        # uses. Naming a run by a setting its recipe ignores makes two
        # different-looking names for the same thing.
        only_under = {
            "decay": "frontier_soft",
            "decay_length": "frontier_soft",
            "signal": "token_signal",
        }

        def name_of(row):
            parts = []
            for axis in varying:
                value = row[axis]
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    continue
                if axis in only_under and row["label"] != only_under[axis]:
                    continue
                parts.append(str(value) if axis == "selection" else f"{axis}={value}")
            return ", ".join(parts) or "base recipe"

        frame["recipe"] = frame.apply(name_of, axis=1)
        return frame[["run", "recipe", "window", "seed"]]

    REPLAY_INDEX = _build_replay_index()

    def recipe_options():
        return sorted(REPLAY_INDEX["recipe"].unique()) if not REPLAY_INDEX.empty else []

    def window_options(recipe):
        """The widths this recipe was actually trained at, newest choice first."""
        rows = REPLAY_INDEX[REPLAY_INDEX["recipe"] == recipe]
        widths = sorted(int(w) for w in rows["window"].dropna().unique())
        # Two of the rules tile or sample rather than placing a window, so they
        # have no width axis at all and are offered a single inert choice.
        return {str(w): w for w in widths} if widths else {"not applicable": -1}

    def sibling_runs(recipe, window):
        """Every seed of one configuration, which is what a spread is read over."""
        rows = REPLAY_INDEX[REPLAY_INDEX["recipe"] == recipe]
        rows = rows[rows["window"].isna()] if window == -1 else rows[rows["window"] == window]
        return rows["run"].tolist()

    def resolve_replay_run(recipe, window, seed):
        rows = REPLAY_INDEX[REPLAY_INDEX["recipe"] == recipe]
        rows = rows[rows["window"].isna()] if window == -1 else rows[rows["window"] == window]
        rows = rows[rows["seed"] == seed]
        return None if rows.empty else rows.iloc[0]["run"]

    def run_picker(recipe):
        """The second half of a run picker, once a recipe has been chosen."""
        widths = window_options(recipe)
        rows = REPLAY_INDEX[REPLAY_INDEX["recipe"] == recipe]
        seeds = {str(int(s)): int(s) for s in sorted(rows["seed"].dropna().unique())} or {
            "none": -1
        }
        return mo.ui.dictionary(
            {
                "window": mo.ui.dropdown(
                    options=widths, value=next(iter(widths)), label="window"
                ),
                "seed": mo.ui.dropdown(
                    options=seeds, value=next(iter(seeds)), label="seed"
                ),
            }
        )

    def recipe_picker():
        options = recipe_options() or ["none"]
        return mo.ui.dropdown(options=options, value=options[0], label="recipe")

    return recipe_picker, resolve_replay_run, run_picker, sibling_runs


@app.cell
def _(mo):
    mo.md("""
    ### Reading a run by eye

    The rule above reports one number per run: a depth, a checkpoint. The three
    figures below exist so that pick can be checked rather than trusted, and so
    the shape behind it stays visible. Each carries its own picker, so they can
    be pointed at three different runs at once and compared.
    """)
    return


@app.cell
def _(recipe_picker):
    warning_recipe = recipe_picker()
    warning_recipe
    return (warning_recipe,)


@app.cell
def _(run_picker, warning_recipe):
    warning_run = run_picker(warning_recipe.value)
    warning_run
    return (warning_run,)


@app.cell
def _(
    FIGURES,
    INK_SOFT,
    RAMP,
    SELECTION_METRIC,
    SERIES,
    missing,
    mo,
    mpl,
    plt,
    replay_frame,
    resolve_replay_run,
    save,
    tidy,
    warning_recipe,
    warning_run,
):
    def trajectory_figure(run):
        """One run's whole training history, at every depth it trained."""
        if run is None:
            return missing("No run was trained at that combination of settings.")
        panel = replay_frame[replay_frame["run"] == run] if not replay_frame.empty else None
        if panel is None or panel.empty:
            return missing("Nothing replayed yet, so there are no histories to draw.")
        layers = sorted(panel["layer"].unique())
        shades = mpl.colors.LinearSegmentedColormap.from_list("depth", RAMP)
        scale = mpl.colors.Normalize(vmin=min(layers), vmax=max(layers))

        fig, axis = plt.subplots(figsize=(6.4, 3.5))
        for layer in layers:
            line = panel[panel["layer"] == layer].sort_values("step")
            axis.plot(
                line["step"],
                line[SELECTION_METRIC],
                color=shades(scale(layer)),
                linewidth=1.2,
            )

        # The depth that ends up being reported is the one whose shape matters, so
        # it is drawn out of the band and its peak marked.
        best_layer = int(panel.loc[panel[SELECTION_METRIC].idxmax(), "layer"])
        chosen = panel[panel["layer"] == best_layer].sort_values("step")
        # Contrasting hue rather than a step of the ramp, so the reported depth
        # reads as a separate thing from the band it came out of.
        axis.plot(chosen["step"], chosen[SELECTION_METRIC], color=SERIES[1], linewidth=2.2)
        top = chosen.loc[chosen[SELECTION_METRIC].idxmax()]
        axis.plot(top["step"], top[SELECTION_METRIC], "o", color=SERIES[1], zorder=5)
        axis.annotate(
            f"layer {best_layer}, step {int(top['step'])}",
            xy=(top["step"], top[SELECTION_METRIC]),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            color=SERIES[1],
            fontsize=8,
        )

        axis.set_xlabel("training step")
        axis.set_ylabel("warning coverage, 256")
        axis.set_title(
            "each line is one depth; shade runs from the first layer to the last",
            color=INK_SOFT,
        )
        tidy(axis, xgrid=False)
        bar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=scale, cmap=shades), ax=axis, pad=0.02
        )
        bar.set_label("layer", color=INK_SOFT)
        bar.outline.set_visible(False)
        fig.suptitle(run[:72], x=0.005, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return mo.vstack(
            [
                save(fig, "replay_trajectories", FIGURES),
                mo.md(
                    "_**Warning coverage** across training: of the tokens in the "
                    "256 before the loop begins, the share this run's probe "
                    "flagged. One line per depth it trained a head at. The depth "
                    "the run would actually be reported at, the one with the best "
                    "warning coverage, is redrawn in colour with its peak "
                    "checkpoint marked; every other depth is context for how much "
                    "depth matters here. The figure directly below is the same "
                    "reading for **in-pattern coverage**, the easy half, so the "
                    "two can be compared depth by depth._"
                ),
            ]
        )

    trajectory_figure(
        resolve_replay_run(
            warning_recipe.value, warning_run.value["window"], warning_run.value["seed"]
        )
    )
    return


@app.cell
def _(replay_frame):
    # `StoppingRule`/`apply_rule_to_run` are already imported (unreturned, but
    # marimo still counts a plain import as a global definition) in the cell
    # above that builds `stopping_outcomes`. Importing under
    # a private, underscore-prefixed alias and wrapping it in a differently
    # named function avoids colliding with that cell while still sharing one
    # rule instance across every plot below.
    from degeneration_probe.evaluation.head_selection import (
        StoppingRule as _StoppingRuleImpl,
        apply_rule_to_run as _apply_rule_to_run_impl,
    )

    DEFAULT_RULE = _StoppingRuleImpl()

    def apply_stopping_rule(history):
        """The shared stopping rule, applied to one run's replayed checkpoints."""
        return _apply_rule_to_run_impl(history, DEFAULT_RULE)

    def run_outcomes(run):
        """What the rule decides for every depth of one replayed run."""
        panel = replay_frame[replay_frame["run"] == run] if not replay_frame.empty else None
        if panel is None or panel.empty:
            return None
        return apply_stopping_rule(panel)

    def best_layer_of(outcomes):
        """The depth the rule would actually report, or None if none qualified."""
        if outcomes is None or outcomes.empty or outcomes["selected_value"].isna().all():
            return None
        return int(outcomes.loc[outcomes["selected_value"].idxmax(), "layer"])

    return DEFAULT_RULE, apply_stopping_rule, best_layer_of, run_outcomes


@app.cell
def _(recipe_picker):
    in_pattern_recipe = recipe_picker()
    in_pattern_recipe
    return (in_pattern_recipe,)


@app.cell
def _(in_pattern_recipe, run_picker):
    in_pattern_run = run_picker(in_pattern_recipe.value)
    in_pattern_run
    return (in_pattern_run,)


@app.cell
def _(
    DEFAULT_RULE,
    FIGURES,
    INK_SOFT,
    RAMP,
    SERIES,
    best_layer_of,
    in_pattern_recipe,
    in_pattern_run,
    missing,
    mo,
    mpl,
    pd,
    plt,
    replay_frame,
    resolve_replay_run,
    run_outcomes,
    save,
    tidy,
):
    def in_pattern_trajectory_figure(run):
        """Every depth's coverage inside the loop, across training.

        This is the half of the objective the rule watches first: a depth has
        to clear the floor here before its warning coverage counts for
        anything, so whether a depth is comfortably above the floor or barely
        scraping it changes how much the pick should be trusted.
        """
        if run is None:
            return missing("No run was trained at that combination of settings.")
        panel = replay_frame[replay_frame["run"] == run] if not replay_frame.empty else None
        if panel is None or panel.empty:
            return missing("Nothing replayed yet, so there is no trajectory to draw.")

        layers = sorted(panel["layer"].unique())
        shades = mpl.colors.LinearSegmentedColormap.from_list("depth", RAMP)
        scale = mpl.colors.Normalize(vmin=min(layers), vmax=max(layers))

        fig, axis = plt.subplots(figsize=(6.4, 3.5))
        for layer in layers:
            line = panel[panel["layer"] == layer].sort_values("step")
            axis.plot(
                line["step"],
                line["in_pattern_recall"],
                color=shades(scale(layer)),
                linewidth=1.2,
            )

        outcomes = run_outcomes(run)
        best_layer = best_layer_of(outcomes)
        if best_layer is not None:
            chosen = panel[panel["layer"] == best_layer].sort_values("step")
            axis.plot(
                chosen["step"], chosen["in_pattern_recall"], color=SERIES[1], linewidth=2.2
            )
            picked = outcomes.loc[outcomes["layer"] == best_layer].iloc[0]
            selected_step = picked["selected_step"]
            if pd.notna(selected_step):
                at_step = chosen[chosen["step"] == int(selected_step)]
                if not at_step.empty:
                    y = float(at_step["in_pattern_recall"].iloc[0])
                    axis.plot(int(selected_step), y, "o", color=SERIES[1], zorder=5)
                    axis.annotate(
                        f"layer {best_layer}, step {int(selected_step)}",
                        xy=(int(selected_step), y),
                        xytext=(-8, 8),
                        textcoords="offset points",
                        ha="right",
                        color=SERIES[1],
                        fontsize=8,
                    )

        axis.axhline(DEFAULT_RULE.floor, color=INK_SOFT, linestyle="--", linewidth=1)
        axis.annotate(
            f"floor to become selectable, {DEFAULT_RULE.floor:g}",
            xy=(panel["step"].max(), DEFAULT_RULE.floor),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
            color=INK_SOFT,
        )

        axis.set_xlabel("training step")
        axis.set_ylabel("coverage inside the loop")
        axis.set_title(
            "each line is one depth; the dashed line is the eligibility floor",
            color=INK_SOFT,
        )
        tidy(axis, xgrid=False)
        bar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=scale, cmap=shades), ax=axis, pad=0.02
        )
        bar.set_label("layer", color=INK_SOFT)
        bar.outline.set_visible(False)
        fig.suptitle(run[:72], x=0.005, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return mo.vstack(
            [
                save(fig, "replay_in_pattern_trajectory", FIGURES),
                mo.md(
                    "_Same reading as the trajectory plot above, but for coverage "
                    "**inside** the loop rather than the run-up before it — this is "
                    "the gate a depth must clear before its warning coverage is "
                    "trusted at all. A depth flat for the second half has "
                    "converged; one still rising was not given enough budget; one "
                    "that falls after an early peak is a third, worse failure mode "
                    "distinct from both. Some real picks in this data sit only a "
                    "few points above the floor — a margin this plot shows "
                    "directly, not just as an eligible/not-eligible flag._"
                ),
            ]
        )

    in_pattern_trajectory_figure(
        resolve_replay_run(
            in_pattern_recipe.value,
            in_pattern_run.value["window"],
            in_pattern_run.value["seed"],
        )
    )
    return


@app.cell
def _(recipe_picker):
    tradeoff_recipe = recipe_picker()
    tradeoff_recipe
    return (tradeoff_recipe,)


@app.cell
def _(mo, run_picker, tradeoff_recipe):
    tradeoff_run = run_picker(tradeoff_recipe.value)
    tradeoff_band = mo.ui.dropdown(
        options={"256 tokens": 256, "128 tokens": 128},
        value="256 tokens",
        label="warning band",
    )
    mo.hstack([tradeoff_run, tradeoff_band], justify="start", gap=2)
    return tradeoff_band, tradeoff_run


@app.cell
def _(
    FIGURES,
    INK_MUTED,
    INK_SOFT,
    RAMP,
    SERIES,
    apply_stopping_rule,
    best_layer_of,
    missing,
    mo,
    mpl,
    pd,
    plt,
    replay_frame,
    resolve_replay_run,
    run_outcomes,
    save,
    sibling_runs,
    tidy,
    tradeoff_band,
    tradeoff_recipe,
    tradeoff_run,
):
    def depth_tradeoff_points(runs, band=256):
        """Both metrics at each depth's own pick, across the seeds of one recipe."""
        if replay_frame.empty or not runs:
            return None
        warning = f"warning_recall_{int(band)}"
        siblings = replay_frame[replay_frame["run"].isin(runs)]
        rows = []
        for seed, seed_panel in siblings.groupby("seed"):
            outcomes = apply_stopping_rule(seed_panel)
            for picked in outcomes.itertuples():
                if pd.isna(picked.selected_step):
                    continue
                at_step = seed_panel[
                    (seed_panel["layer"] == picked.layer)
                    & (seed_panel["step"] == picked.selected_step)
                ]
                if at_step.empty:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "layer": picked.layer,
                        "warning": float(at_step[warning].iloc[0]),
                        "in_pattern_recall": float(at_step["in_pattern_recall"].iloc[0]),
                    }
                )
        return pd.DataFrame(rows) if rows else None

    def coverage_tradeoff_figure(runs, run, band):
        long = depth_tradeoff_points(runs, band)
        if long is None:
            return missing(
                "No depth of this run's configuration has become selectable yet."
            )
        best_layer = best_layer_of(run_outcomes(run)) if run else None

        grouped = (
            long.groupby("layer")
            .agg(
                warning_mean=("warning", "mean"),
                in_pattern_mean=("in_pattern_recall", "mean"),
            )
            .reset_index()
            .sort_values("layer")
        )

        fig, axis = plt.subplots(figsize=(6.0, 4.0))
        shades = mpl.colors.LinearSegmentedColormap.from_list("depth", RAMP)
        scale = mpl.colors.Normalize(
            vmin=grouped["layer"].min(), vmax=grouped["layer"].max()
        )
        axis.plot(
            grouped["in_pattern_mean"], grouped["warning_mean"],
            color=INK_MUTED, linewidth=0.8, zorder=1,
        )
        axis.scatter(
            grouped["in_pattern_mean"], grouped["warning_mean"],
            c=grouped["layer"], cmap=shades, norm=scale, zorder=2, s=34,
            edgecolors="white", linewidths=0.6,
        )
        if best_layer is not None and best_layer in set(grouped["layer"]):
            top = grouped[grouped["layer"] == best_layer].iloc[0]
            axis.plot(
                top["in_pattern_mean"], top["warning_mean"], "o",
                color=SERIES[1], markersize=12, markerfacecolor="none",
                markeredgewidth=2, zorder=3,
            )
            axis.annotate(
                f"layer {best_layer}",
                xy=(top["in_pattern_mean"], top["warning_mean"]),
                xytext=(-8, 8), textcoords="offset points", fontsize=8,
                color=SERIES[1], ha="right",
            )
        axis.margins(0.12)
        axis.set_xlabel("in-pattern coverage")
        axis.set_ylabel(f"warning coverage, {int(band)}")
        axis.set_title("one point per depth, at that depth's own pick", color=INK_SOFT)
        bar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=scale, cmap=shades), ax=axis, pad=0.02
        )
        bar.set_label("layer", color=INK_SOFT)
        bar.outline.set_visible(False)
        tidy(axis, xgrid=False)

        fig.suptitle(
            "Would giving up coverage buy a more trustworthy pick?",
            x=0.005, ha="left", fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return mo.vstack(
            [
                save(fig, "replay_coverage_tradeoff", FIGURES),
                mo.md(
                    "_Every depth of this configuration, each read at its own "
                    "selected checkpoint rather than only the winning depth's. "
                    "The two coverage numbers are plotted against each other, "
                    "one point per depth, connected in depth order and coloured "
                    "by the same depth ramp so the direction is legible. A depth "
                    "up and to the right of another is strictly better on both "
                    "counts._\n\n"
                    "_**The dots and the ring are not the same population, and "
                    "they can disagree.** Each dot is the mean across every seed "
                    "of the recipe. The ring is the depth the rule picks for the "
                    "one seed chosen above, and the rule runs per seed. Seeds do "
                    "not always agree on a depth: on the plain `all_tokens` "
                    "recipe, for instance, seed 42 picks layer 15 while seeds 43 "
                    "and 44 pick layer 12. So with seed 42 selected the ring "
                    "lands on 15 even though the seed-averaged dots peak at 12. "
                    "That gap is a statement about how stable the pick is, which "
                    "is the thing worth seeing here; change the seed and watch "
                    "whether the ring moves._"
                ),
            ]
        )

    coverage_tradeoff_figure(
        sibling_runs(tradeoff_recipe.value, tradeoff_run.value["window"]),
        resolve_replay_run(
            tradeoff_recipe.value,
            tradeoff_run.value["window"],
            tradeoff_run.value["seed"],
        ),
        tradeoff_band.value,
    )
    return (depth_tradeoff_points,)


@app.cell
def _(mo):
    mo.md("""
    ---
    # Part 2. The experiments

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
            "Only the settings this stage actually changes are listed below, "
            "with every value it takes; a setting that's fixed within this stage "
            "isn't shown, even if it differs from the shared default.\n\n"
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

    def _has_any_scoring(run_dir, own_layer, split):
        root = Path(run_dir)
        if any(root.glob(f"layers/layer_*/evaluation/{split}")):
            return True
        own = own_depth(own_layer)
        return own is not None and (root / "evaluation" / split).is_dir()

    def scoring_progress(exp_id, split):
        """How many of this stage's finished runs have any scoring at all on this split.

        Deliberately coarser than ``scored_depths``: this counts a run once it
        has been scored at any depth, so a table or figure showing only a
        handful of rows can say plainly how much of the stage that is, rather
        than leaving a reader to wonder whether the rest never trained.
        """
        rows = finished_rows(exp_id)
        if rows.empty:
            return 0, 0
        scored = sum(
            _has_any_scoring(row.run_dir, row.layer, split) for row in rows.itertuples()
        )
        return scored, len(rows)

    return (
        load_baseline_view,
        load_view,
        nothing_scored,
        scored_depths,
        scoring_progress,
    )


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

        Model-free scorers ride along only in S0, where they are the whole
        result, and in S4, the one place a probe is measured against the floor
        again on data nothing was tuned on. Every stage in between compares
        probe configurations against each other; S0 already settled how they
        compare to the floor, so repeating that comparison in every stage would
        only add rows without adding a question.
        """
        records = _probe_records(exp_id, split, layer, budget)
        if exp_id == "S0" or (records and exp_id == "S4"):
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
def _(
    METRIC_ORDER,
    is_baseline,
    mo,
    nothing_scored,
    scorecard_long,
    scoring_progress,
):
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

        # A depth only means something once a probe is in the picture: the
        # model-free scorers read no layer at all, so a table of nothing but
        # baselines should never claim to be read "at layer N".
        only_baselines = all(is_baseline(name) for name in table["configuration"])
        where = f"`{split}`" if only_baselines else f"`{split}`, layer {layer}"
        has_baseline = any(is_baseline(name) for name in table["configuration"])
        baseline_note = (
            " Model-free scorers are the first rows, shown here for reference."
            if has_baseline and not only_baselines
            else ""
        )
        scored, finished = scoring_progress(exp_id, split)
        progress_note = (
            f" **{scored} of {finished}** finished runs of this stage have been "
            "through the evaluator on this split so far — the rest are trained "
            "but not yet scored, and simply have no row here yet."
            if finished and scored < finished
            else ""
        )
        return mo.vstack(
            [
                mo.md(
                    "One row per configuration: the median across seeds, with the "
                    "seed-to-seed range in brackets where it varies. At a "
                    f"**{budget:.0%} false-alarm budget** on {where}."
                    f"{baseline_note}{progress_note}"
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
    mo,
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
        return mo.vstack(
            [
                save(fig, f"{exp_id}_{split}_L{int(layer):02d}_views", FIGURES),
                mo.md(
                    "_Four panels, one per view: **A** is whether it fires on the "
                    "right answers at all (recall, precision); **B** is what share "
                    "of tokens it flags, split between the easy tokens already "
                    "inside the loop and the harder run-up just before it; **C** is "
                    "how early or late the alarm lands relative to the true onset "
                    "(negative is early, the vertical line marks zero); **D** is "
                    "how long the first alarm holds once it fires. Each row is one "
                    "configuration; the dot is the median across seeds and the bar "
                    "is the seed-to-seed range._"
                ),
            ]
        )

    return (views_figure,)


@app.cell
def _(
    BY_ID,
    FIGURES,
    INK_SOFT,
    PROBED_LAYERS,
    Path,
    RAMP,
    config_of,
    finished_rows,
    missing,
    mo,
    mpl,
    pd,
    plt,
    save,
    tidy,
):
    ALL_CONFIGURATIONS = "every configuration"

    def loss_configurations(exp_id):
        """The configurations of a stage that logged a validation curve."""
        rows = finished_rows(exp_id)
        if rows.empty:
            return []
        axes = BY_ID[exp_id]["axes"]
        names = {
            config_of(row, axes)
            for row in rows.itertuples()
            if (Path(row.run_dir) / "history.parquet").is_file()
        }
        return sorted(names)

    def loss_config_control(exp_id):
        options = [ALL_CONFIGURATIONS] + loss_configurations(exp_id)
        return mo.ui.dropdown(
            options=options, value=options[0], label="configuration"
        )

    def stage_curves(exp_id, layers, configuration=ALL_CONFIGURATIONS):
        """Validation loss and score spread, pooled over a stage's finished runs."""
        rows = finished_rows(exp_id)
        if rows.empty:
            return pd.DataFrame()
        axes = BY_ID[exp_id]["axes"]
        frames = []
        for row in rows.itertuples():
            if configuration != ALL_CONFIGURATIONS and config_of(row, axes) != configuration:
                continue
            path = Path(row.run_dir) / "history.parquet"
            if not path.is_file():
                continue
            history = pd.read_parquet(path)
            for depth in layers:
                loss_column = f"val/layer{int(depth):02d}/loss_unweighted"
                spread_column = f"val/layer{int(depth):02d}/prediction_std"
                if loss_column not in history.columns:
                    continue
                wanted = {loss_column: "loss"}
                if spread_column in history.columns:
                    wanted[spread_column] = "spread"
                piece = history[["step"] + list(wanted)].rename(columns=wanted)
                piece["layer"] = int(depth)
                piece["seed"] = row.seed
                piece["run"] = Path(row.run_dir).parent.name
                frames.append(piece.dropna(subset=["loss"]))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def curve_figure(exp_id, layer, configuration=ALL_CONFIGURATIONS):
        """Is the run healthy, and did it have enough budget to settle?

        ``layer`` is a depth, or the string ``all`` for every depth at once.
        """
        every_depth = layer == "all"
        layers = list(PROBED_LAYERS) if every_depth else [int(layer)]
        curves = stage_curves(exp_id, layers, configuration)
        if curves.empty:
            return missing(
                f"No finished run for {exp_id} logged a validation curve at "
                + ("any depth." if every_depth else f"layer {layer}.")
            )
        run_count = curves["run"].nunique()

        panels = [
            ("loss", "validation loss, class weight removed"),
            ("spread", "spread of the scores"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
        shades = mpl.colors.LinearSegmentedColormap.from_list("depth", RAMP)
        drawn = sorted(curves["layer"].unique())
        scale = mpl.colors.Normalize(vmin=min(drawn), vmax=max(drawn))
        for axis, (column, title) in zip(axes, panels):
            if column not in curves.columns:
                axis.set_axis_off()
                continue
            if every_depth:
                # One line per depth, median across the stage's runs and seeds.
                # A band per depth would be thirty-one overlapping bands, which
                # hides the depth ordering this view exists to show.
                for depth in drawn:
                    line = (
                        curves[curves["layer"] == depth]
                        .groupby("step")[column]
                        .median()
                    )
                    axis.plot(
                        line.index,
                        line.to_numpy(),
                        color=shades(scale(depth)),
                        linewidth=1.0,
                    )
            else:
                band = curves.groupby("step")[column].agg(["min", "median", "max"])
                axis.fill_between(
                    band.index, band["min"], band["max"],
                    color=RAMP[0], alpha=0.6, linewidth=0,
                )
                axis.plot(band.index, band["median"], color=RAMP[3])
            if column == "spread":
                axis.axhline(0.01, color="#e34948", linestyle="--", linewidth=1)
            axis.set_title(title, color=INK_SOFT)
            axis.set_xlabel("step")
            tidy(axis, xgrid=False)
        where = "every depth" if every_depth else f"layer {layer}"
        which = (
            f"all {run_count} runs of the stage"
            if configuration == ALL_CONFIGURATIONS
            else f"{configuration} ({run_count} seed runs)"
        )
        fig.suptitle(
            f"{exp_id}, {where}: {which}",
            x=0.005,
            ha="left",
            fontsize=10,
        )
        # One colour bar for two panels has to be placed by hand: asking for it
        # across both axes leaves the figure impossible to lay out automatically.
        if every_depth:
            fig.tight_layout(rect=(0, 0, 0.9, 0.9))
            bar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=scale, cmap=shades),
                cax=fig.add_axes((0.925, 0.14, 0.015, 0.66)),
            )
            bar.set_label("layer", color=INK_SOFT)
            bar.outline.set_visible(False)
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.9))
        stem = "Lall" if every_depth else f"L{int(layer):02d}"
        return mo.vstack(
            [
                save(fig, f"{exp_id}_{stem}_curves", FIGURES),
                mo.md(
                    "_Not a result, a health check: does the run look like it "
                    "trained properly. Left is validation loss with the class "
                    "weight removed, so different recipes sit on the same scale. "
                    "Right is the spread (standard deviation) of the probe's own "
                    "scores; the dashed red line is the collapse threshold, and a "
                    "run that falls to it is emitting close to a constant score for "
                    "every token, which can still show a plausible loss while "
                    "telling positives and negatives apart not at all._\n\n"
                    "_**This is never one run.** It pools every run named in the "
                    "title: at a single depth the line is the median over them "
                    "with the full range shaded, and at every depth it is one "
                    "median line per depth, so a depth that trains differently "
                    "from its neighbours shows as a line out of the ramp. Narrow "
                    "it with the configuration picker above to see one recipe's "
                    "three seeds instead of the whole stage._"
                ),
            ]
        )

    return curve_figure, loss_config_control


@app.cell
def _(DEFAULT_LAYER, PROBED_LAYERS, mo):
    # Every figure and table below builds its own pickers from these, so two of
    # them can be read at different depths or budgets at the same time. Nothing
    # here is a shared setting: each call returns fresh widgets belonging to the
    # one cell that displays them.
    SPLITS = ["val", "test_indomain", "test_heldout_domains"]
    BUDGETS = {"1%": 0.01, "5%": 0.05, "10%": 0.10}

    def split_control(value="val", *, options=None):
        choices = list(options) if options else SPLITS
        return mo.ui.dropdown(options=choices, value=value, label="split")

    TEST_SPLITS = ["test_indomain", "test_heldout_domains"]

    def test_view_controls():
        """As `view_controls`, but on the held-out splits only.

        S4 is the one place validation must not be selectable: it is the whole
        point of the stage that nothing there was chosen on the data it reports.
        """
        return mo.ui.dictionary(
            {
                "split": split_control("test_indomain", options=TEST_SPLITS),
                "layer": layer_control(),
                "budget": budget_control(),
            }
        )

    def layer_control(value=DEFAULT_LAYER, *, with_all=False):
        options = {str(n): n for n in PROBED_LAYERS}
        if with_all:
            options = {"every depth": "all", **options}
        key = "every depth" if value == "all" else str(value)
        return mo.ui.dropdown(options=options, value=key, label="depth")

    def budget_control(value="1%"):
        return mo.ui.dropdown(options=BUDGETS, value=value, label="false-alarm budget")

    def view_controls():
        """The split, depth and budget one scored table or figure is read at."""
        return mo.ui.dictionary(
            {
                "split": split_control(),
                "layer": layer_control(),
                "budget": budget_control(),
            }
        )

    return (
        budget_control,
        layer_control,
        split_control,
        test_view_controls,
        view_controls,
    )


@app.cell
def _(BY_ID, Path, all_runs, config_of, mo, pd, replay_frame):
    # The two coverage numbers come from the replayed checkpoints rather than
    # from the scored runs. Full scoring is affordable only at the single depth
    # a run is reported at, so a stage's scored runs hold one depth each and
    # cannot answer a question asked per depth. The replay holds all thirty-one,
    # at every saved step, on the same pinned validation population, which is
    # what both figures below need.
    def stage_replay(exp_id):
        """Every replayed checkpoint of a stage's runs, named by configuration."""
        if replay_frame.empty or all_runs.empty:
            return pd.DataFrame()
        axes = BY_ID[exp_id]["axes"]
        naming = {}
        for row in all_runs.itertuples():
            if exp_id not in (row.exps or []):
                continue
            naming[Path(row.run_dir).parent.name] = config_of(row, axes)
        if not naming:
            return pd.DataFrame()
        panel = replay_frame[replay_frame["run"].isin(naming)].copy()
        if panel.empty:
            return panel
        panel["configuration"] = panel["run"].map(naming)
        return panel

    def stage_configurations(exp_id):
        panel = stage_replay(exp_id)
        return [] if panel.empty else sorted(panel["configuration"].unique())

    def config_control(exp_id):
        """This stage's own configuration picker, for one figure."""
        options = stage_configurations(exp_id) or ["none"]
        return mo.ui.dropdown(options=options, value=options[0], label="configuration")

    return config_control, stage_replay


@app.cell
def _(
    FIGURES,
    INK_MUTED,
    INK_SOFT,
    RAMP,
    SERIES,
    depth_tradeoff_points,
    missing,
    mo,
    mpl,
    plt,
    save,
    stage_replay,
    tidy,
):
    def stage_tradeoff_figure(exp_id, configuration):
        """The easy half against the hard half, one point per depth."""
        panel = stage_replay(exp_id)
        if panel.empty:
            return missing(
                f"No run of {exp_id} has had its checkpoints replayed, so the two "
                "coverage numbers cannot be read per depth."
            )
        runs = panel.loc[panel["configuration"] == configuration, "run"].unique().tolist()
        points = depth_tradeoff_points(runs)
        if points is None:
            return missing("No depth of this configuration ever became selectable.")
        grouped = (
            points.groupby("layer")
            .agg(
                warning=("warning", "mean"),
                in_pattern=("in_pattern_recall", "mean"),
            )
            .reset_index()
            .sort_values("layer")
        )

        fig, axis = plt.subplots(figsize=(6.0, 4.0))
        shades = mpl.colors.LinearSegmentedColormap.from_list("depth", RAMP)
        scale = mpl.colors.Normalize(
            vmin=grouped["layer"].min(), vmax=grouped["layer"].max()
        )
        axis.plot(
            grouped["in_pattern"], grouped["warning"],
            color=INK_MUTED, linewidth=0.8, zorder=1,
        )
        axis.scatter(
            grouped["in_pattern"], grouped["warning"],
            c=grouped["layer"], cmap=shades, norm=scale, s=34, zorder=2,
            edgecolors="white", linewidths=0.6,
        )
        best = grouped.loc[grouped["warning"].idxmax()]
        axis.annotate(
            f"layer {int(best['layer'])}",
            xy=(best["in_pattern"], best["warning"]),
            xytext=(-8, 8), textcoords="offset points", fontsize=8,
            color=SERIES[1], ha="right",
        )
        axis.margins(0.12)
        axis.set_xlabel("in-pattern coverage")
        axis.set_ylabel("warning coverage, 256")
        axis.set_title(configuration[:60], color=INK_SOFT)
        bar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=scale, cmap=shades), ax=axis, pad=0.02
        )
        bar.set_label("layer", color=INK_SOFT)
        bar.outline.set_visible(False)
        tidy(axis, xgrid=False)
        fig.suptitle(
            f"{exp_id}: where this configuration's depths sit on the trade-off",
            x=0.005, ha="left", fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return mo.vstack(
            [
                save(fig, f"{exp_id}_stage_tradeoff", FIGURES),
                mo.md(
                    "_One point per depth of the selected configuration, each read "
                    "at its own selected checkpoint and averaged over the "
                    "configuration's seeds, at the 1% false-alarm budget the replay "
                    "was measured at. The horizontal axis is the easy half, tokens "
                    "already inside the loop; the vertical axis is the half this "
                    "project is about, tokens in the 256 before it. A depth up and "
                    "to the right of another is better on both counts, and a stage "
                    "whose depths run diagonally down to the right is buying the "
                    "run-up by giving up the loop._"
                ),
            ]
        )

    def stage_tradeoff_trajectory(exp_id, configuration, layer):
        """The same trade-off for one depth, traced across training."""
        panel = stage_replay(exp_id)
        if panel.empty:
            return missing(
                f"No run of {exp_id} has had its checkpoints replayed, so there is "
                "no trajectory to trace."
            )
        chosen = panel[
            (panel["configuration"] == configuration) & (panel["layer"] == int(layer))
        ]
        if chosen.empty:
            return missing(f"This configuration has no replayed history at layer {layer}.")
        line = (
            chosen.groupby("step")[["in_pattern_recall", "warning_recall_256"]]
            .median()
            .reset_index()
            .sort_values("step")
        )

        fig, axis = plt.subplots(figsize=(6.0, 4.0))
        steps = mpl.colors.LinearSegmentedColormap.from_list("step", RAMP)
        scale = mpl.colors.Normalize(vmin=line["step"].min(), vmax=line["step"].max())
        axis.plot(
            line["in_pattern_recall"], line["warning_recall_256"],
            color=INK_MUTED, linewidth=0.8, zorder=1,
        )
        axis.scatter(
            line["in_pattern_recall"], line["warning_recall_256"],
            c=line["step"], cmap=steps, norm=scale, s=30, zorder=2,
            edgecolors="white", linewidths=0.6,
        )
        # The path usually ends at the right-hand edge, so its label is written
        # back into the plot rather than out toward the colour bar.
        for marker, row, offset, side in (
            ("start", line.iloc[0], (6, -10), "left"),
            ("end", line.iloc[-1], (-6, 8), "right"),
        ):
            axis.annotate(
                f"{marker}, step {int(row['step'])}",
                xy=(row["in_pattern_recall"], row["warning_recall_256"]),
                xytext=offset,
                textcoords="offset points", fontsize=8, color=INK_SOFT, ha=side,
            )
        axis.margins(0.12)
        axis.set_xlabel("in-pattern coverage")
        axis.set_ylabel("warning coverage, 256")
        axis.set_title(f"{configuration[:52]}, layer {layer}", color=INK_SOFT)
        bar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=scale, cmap=steps), ax=axis, pad=0.02
        )
        bar.set_label("training step", color=INK_SOFT)
        bar.outline.set_visible(False)
        tidy(axis, xgrid=False)
        fig.suptitle(
            f"{exp_id}: how one depth moves across the trade-off while it trains",
            x=0.005, ha="left", fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return mo.vstack(
            [
                save(fig, f"{exp_id}_stage_tradeoff_steps", FIGURES),
                mo.md(
                    "_The same two axes as the figure above, but one depth traced "
                    "through training instead of every depth at one moment, median "
                    "across the configuration's seeds and coloured by step. A path "
                    "that climbs up and to the right is a depth still learning both "
                    "halves. One that runs right while flattening has stopped "
                    "improving on the run-up and is only getting better at the loop, "
                    "which is the point at which more steps stop being worth their "
                    "cost._"
                ),
            ]
        )

    return stage_tradeoff_figure, stage_tradeoff_trajectory


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
def _(mo):
    mo.md("""
    **Before the table:** `repetition` is the only one of these three that can
    actually spend a false-alarm budget. `lrs` and `entropy` both score an
    answer by its single highest-scoring token, and almost every answer has
    one: 81% of healthy answers tie at the top of `lrs`'s range, 98.6% for
    `entropy`. A 1% budget cannot break a tie that size, so both scorers get
    pushed to a threshold above their whole range and fire on nothing,
    positives included. Their rows below read zero everywhere on purpose —
    that is not a bug and not a sign they are secretly flawless, it means they
    need a different aggregation before they can compete on this table at all.
    """)
    return


@app.cell
def _(view_controls):
    s0_card_controls = view_controls()
    s0_card_controls
    return (s0_card_controls,)


@app.cell
def _(s0_card_controls, scorecard):
    scorecard(
        "S0",
        s0_card_controls.value["split"],
        s0_card_controls.value["layer"],
        s0_card_controls.value["budget"],
    )
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
    next section is about why.
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
    posts, its lead time and its warning coverage, is therefore an upper
    bound that no deployment could reach.

    The size of that gap is worth knowing, so it is measured below rather than
    argued about. The honest version of the same statistic uses the window
    *behind* the token, $[t - 256,\ t)$, and that is exactly the value already
    stored 256 positions earlier, so it needs no recomputation: shifting the
    scores forward by the window size gives the score a live system could have
    had.

    **It's fine for labeling.** This only becomes important if we try to compare
    the probe's performance with these heuristics. By comparing a probe with the repetition
    score that can also look at the window after token t, would not be a fair comparison.
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
                    collapses: warning coverage falls by about four fifths,
                    and the median alarm moves from before the loop to well
                    inside it. The apparent head start was the window length.

                    Read against this, a probe does not have to beat 0.15
                    warning coverage, it has to beat roughly 0.04, and it
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
       aimed at spans that already look repetitive. Adds difficulty. (we should also make some test to see how the probes perform on sentences that look repetitive, but are not degenerate, like math generations. still not done a test on this.)

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
def _(layer_control, loss_config_control, mo):
    s1_loss_depth = layer_control(value="all", with_all=True)
    s1_loss_config = loss_config_control("S1")
    mo.hstack([s1_loss_config, s1_loss_depth], justify="start", gap=2)
    return s1_loss_config, s1_loss_depth


@app.cell
def _(curve_figure, s1_loss_config, s1_loss_depth):
    curve_figure("S1", s1_loss_depth.value, s1_loss_config.value)
    return


@app.cell
def _(view_controls):
    s1_card_controls = view_controls()
    s1_card_controls
    return (s1_card_controls,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [why below only frontier_window, window=128 is shown? why not all of them? even if you were only showing runs that got scored, what about the others? why not still show the results of the 400-ish validtion row that were scored? that results could still be useful, right? maybe we could have a part for just the intermediate scores and one part for the whole scoring. we could have a table where we can select the evalution steps to look for or also to look at the best eval step for all training strategy]
    """)
    return


@app.cell
def _(s1_card_controls, scorecard):
    scorecard(
        "S1",
        s1_card_controls.value["split"],
        s1_card_controls.value["layer"],
        s1_card_controls.value["budget"],
    )
    return


@app.cell
def _(view_controls):
    s1_views_controls = view_controls()
    s1_views_controls
    return (s1_views_controls,)


@app.cell
def _(s1_views_controls, views_figure):
    views_figure(
        "S1",
        s1_views_controls.value["split"],
        s1_views_controls.value["layer"],
        s1_views_controls.value["budget"],
    )
    return


@app.cell
def _(budget_control, mo, split_control):
    s1_depth_split = split_control()
    s1_depth_budget = budget_control()
    mo.hstack([s1_depth_split, s1_depth_budget], justify="start", gap=2)
    return s1_depth_budget, s1_depth_split


@app.cell
def _(
    FIGURES,
    INK_SOFT,
    SERIES,
    finished_rows,
    load_view,
    missing,
    mo,
    pd,
    plt,
    s1_depth_budget,
    s1_depth_split,
    save,
    scored_depths,
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
                # Markers, not a bare line: a rule scored at a single depth is
                # one point, and a line through one point draws nothing at all.
                axis.plot(
                    line.index, line.to_numpy(), "o-", color=colour, label=rule,
                    markersize=4,
                )
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
        return mo.vstack(
            [
                save(fig, f"{exp_id}_{split}_depth", FIGURES),
                mo.md(
                    "_Where in the network the approach becomes visible: warning "
                    "coverage over the 256 tokens before the frontier, at every "
                    "**fully scored** depth, one line per selection rule, one "
                    "panel per window size. Only runs that have been through the "
                    "evaluator can appear, and most have been scored at a single "
                    "depth, so a panel showing isolated dots rather than curves "
                    "means exactly one depth was scored for that rule. The W=256 "
                    "panel is the extreme case: `frontier_window` at that width "
                    "was scored at layer 15 and nowhere else, so it is one dot "
                    "and there is no profile to draw yet._"
                ),
            ]
        )

    depth_figure("S1", s1_depth_split.value, s1_depth_budget.value)
    return


@app.cell
def _(config_control):
    s1_tradeoff_config = config_control("S1")
    s1_tradeoff_config
    return (s1_tradeoff_config,)


@app.cell
def _(s1_tradeoff_config, stage_tradeoff_figure):
    stage_tradeoff_figure("S1", s1_tradeoff_config.value)
    return


@app.cell
def _(config_control, layer_control, mo):
    s1_steps_config = config_control("S1")
    s1_steps_depth = layer_control()
    mo.hstack([s1_steps_config, s1_steps_depth], justify="start", gap=2)
    return s1_steps_config, s1_steps_depth


@app.cell
def _(s1_steps_config, s1_steps_depth, stage_tradeoff_trajectory):
    stage_tradeoff_trajectory(
        "S1", s1_steps_config.value, s1_steps_depth.value
    )
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
def _(layer_control, loss_config_control, mo):
    s2a_loss_depth = layer_control(value="all", with_all=True)
    s2a_loss_config = loss_config_control("S2a")
    mo.hstack([s2a_loss_config, s2a_loss_depth], justify="start", gap=2)
    return s2a_loss_config, s2a_loss_depth


@app.cell
def _(curve_figure, s2a_loss_config, s2a_loss_depth):
    curve_figure("S2a", s2a_loss_depth.value, s2a_loss_config.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [here as above: why are we only showing the checkpoints that got scored? can't we show in a table all of them and then on a separate part only the three that got scored? otherwise we are loosing the results of all the other runs. this apply also for the following experimental sections]
    """)
    return


@app.cell
def _(view_controls):
    s2a_card_controls = view_controls()
    s2a_card_controls
    return (s2a_card_controls,)


@app.cell
def _(s2a_card_controls, scorecard):
    scorecard(
        "S2a",
        s2a_card_controls.value["split"],
        s2a_card_controls.value["layer"],
        s2a_card_controls.value["budget"],
    )
    return


@app.cell
def _(view_controls):
    s2a_horizon_controls = view_controls()
    s2a_horizon_controls
    return (s2a_horizon_controls,)


@app.cell
def _(
    FIGURES,
    INK_MUTED,
    INK_SOFT,
    SERIES,
    finished_rows,
    load_view,
    missing,
    mo,
    pd,
    plt,
    s2a_horizon_controls,
    save,
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
        return mo.vstack(
            [
                save(fig, f"S2a_{split}_L{int(layer):02d}_horizon", FIGURES),
                mo.md(
                    "_What moving the label boundary earlier actually buys, against "
                    "the horizon length in tokens: warning coverage (left), median "
                    "lead time with zero marking the true onset (middle), and how "
                    "many degenerate answers are never flagged at all (right). If a "
                    "longer horizon doesn't move these, the probe isn't being "
                    "taught to fire earlier, whatever the label says it should be "
                    "doing._"
                ),
            ]
        )

    horizon_figure(
        s2a_horizon_controls.value["split"],
        s2a_horizon_controls.value["layer"],
        s2a_horizon_controls.value["budget"],
    )
    return


@app.cell
def _(config_control):
    s2a_tradeoff_config = config_control("S2a")
    s2a_tradeoff_config
    return (s2a_tradeoff_config,)


@app.cell
def _(s2a_tradeoff_config, stage_tradeoff_figure):
    stage_tradeoff_figure("S2a", s2a_tradeoff_config.value)
    return


@app.cell
def _(config_control, layer_control, mo):
    s2a_steps_config = config_control("S2a")
    s2a_steps_depth = layer_control()
    mo.hstack([s2a_steps_config, s2a_steps_depth], justify="start", gap=2)
    return s2a_steps_config, s2a_steps_depth


@app.cell
def _(s2a_steps_config, s2a_steps_depth, stage_tradeoff_trajectory):
    stage_tradeoff_trajectory(
        "S2a", s2a_steps_config.value, s2a_steps_depth.value
    )
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
def _(layer_control, loss_config_control, mo):
    s2b_loss_depth = layer_control(value="all", with_all=True)
    s2b_loss_config = loss_config_control("S2b")
    mo.hstack([s2b_loss_config, s2b_loss_depth], justify="start", gap=2)
    return s2b_loss_config, s2b_loss_depth


@app.cell
def _(curve_figure, s2b_loss_config, s2b_loss_depth):
    curve_figure("S2b", s2b_loss_depth.value, s2b_loss_config.value)
    return


@app.cell
def _(view_controls):
    s2b_card_controls = view_controls()
    s2b_card_controls
    return (s2b_card_controls,)


@app.cell
def _(s2b_card_controls, scorecard):
    scorecard(
        "S2b",
        s2b_card_controls.value["split"],
        s2b_card_controls.value["layer"],
        s2b_card_controls.value["budget"],
    )
    return


@app.cell
def _(view_controls):
    s2b_views_controls = view_controls()
    s2b_views_controls
    return (s2b_views_controls,)


@app.cell
def _(s2b_views_controls, views_figure):
    views_figure(
        "S2b",
        s2b_views_controls.value["split"],
        s2b_views_controls.value["layer"],
        s2b_views_controls.value["budget"],
    )
    return


@app.cell
def _(config_control):
    s2b_tradeoff_config = config_control("S2b")
    s2b_tradeoff_config
    return (s2b_tradeoff_config,)


@app.cell
def _(s2b_tradeoff_config, stage_tradeoff_figure):
    stage_tradeoff_figure("S2b", s2b_tradeoff_config.value)
    return


@app.cell
def _(config_control, layer_control, mo):
    s2b_steps_config = config_control("S2b")
    s2b_steps_depth = layer_control()
    mo.hstack([s2b_steps_config, s2b_steps_depth], justify="start", gap=2)
    return s2b_steps_config, s2b_steps_depth


@app.cell
def _(s2b_steps_config, s2b_steps_depth, stage_tradeoff_trajectory):
    stage_tradeoff_trajectory(
        "S2b", s2b_steps_config.value, s2b_steps_depth.value
    )
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
def _(layer_control, loss_config_control, mo):
    s2c_loss_depth = layer_control(value="all", with_all=True)
    s2c_loss_config = loss_config_control("S2c")
    mo.hstack([s2c_loss_config, s2c_loss_depth], justify="start", gap=2)
    return s2c_loss_config, s2c_loss_depth


@app.cell
def _(curve_figure, s2c_loss_config, s2c_loss_depth):
    curve_figure("S2c", s2c_loss_depth.value, s2c_loss_config.value)
    return


@app.cell
def _(view_controls):
    s2c_card_controls = view_controls()
    s2c_card_controls
    return (s2c_card_controls,)


@app.cell
def _(s2c_card_controls, scorecard):
    scorecard(
        "S2c",
        s2c_card_controls.value["split"],
        s2c_card_controls.value["layer"],
        s2c_card_controls.value["budget"],
    )
    return


@app.cell
def _(config_control):
    s2c_tradeoff_config = config_control("S2c")
    s2c_tradeoff_config
    return (s2c_tradeoff_config,)


@app.cell
def _(s2c_tradeoff_config, stage_tradeoff_figure):
    stage_tradeoff_figure("S2c", s2c_tradeoff_config.value)
    return


@app.cell
def _(config_control, layer_control, mo):
    s2c_steps_config = config_control("S2c")
    s2c_steps_depth = layer_control()
    mo.hstack([s2c_steps_config, s2c_steps_depth], justify="start", gap=2)
    return s2c_steps_config, s2c_steps_depth


@app.cell
def _(s2c_steps_config, s2c_steps_depth, stage_tradeoff_trajectory):
    stage_tradeoff_trajectory(
        "S2c", s2c_steps_config.value, s2c_steps_depth.value
    )
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
def _(layer_control, loss_config_control, mo):
    s2d_loss_depth = layer_control(value="all", with_all=True)
    s2d_loss_config = loss_config_control("S2d")
    mo.hstack([s2d_loss_config, s2d_loss_depth], justify="start", gap=2)
    return s2d_loss_config, s2d_loss_depth


@app.cell
def _(curve_figure, s2d_loss_config, s2d_loss_depth):
    curve_figure("S2d", s2d_loss_depth.value, s2d_loss_config.value)
    return


@app.cell
def _(view_controls):
    s2d_card_controls = view_controls()
    s2d_card_controls
    return (s2d_card_controls,)


@app.cell
def _(s2d_card_controls, scorecard):
    scorecard(
        "S2d",
        s2d_card_controls.value["split"],
        s2d_card_controls.value["layer"],
        s2d_card_controls.value["budget"],
    )
    return


@app.cell
def _(config_control):
    s2d_tradeoff_config = config_control("S2d")
    s2d_tradeoff_config
    return (s2d_tradeoff_config,)


@app.cell
def _(s2d_tradeoff_config, stage_tradeoff_figure):
    stage_tradeoff_figure("S2d", s2d_tradeoff_config.value)
    return


@app.cell
def _(config_control, layer_control, mo):
    s2d_steps_config = config_control("S2d")
    s2d_steps_depth = layer_control()
    mo.hstack([s2d_steps_config, s2d_steps_depth], justify="start", gap=2)
    return s2d_steps_config, s2d_steps_depth


@app.cell
def _(s2d_steps_config, s2d_steps_depth, stage_tradeoff_trajectory):
    stage_tradeoff_trajectory(
        "S2d", s2d_steps_config.value, s2d_steps_depth.value
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Where things stand: the leading candidate from each stage

    Every stage above compares its own recipes against each other. This section
    is the first place results from different stages sit side by side: the
    leading candidate out of S1, out of S2a, out of S2b and out of S2d, each
    scored by the full protocol rather than read off the cheaper estimate that
    picked them. S2c is not represented — it was ruled out earlier without a
    full scoring pass, so there is no candidate from it to compare here.

    Twelve configurations, three seeds each, all scored at the single depth
    each was picked at. These twelve are also what S3's adapted runs are built
    from, one adapter per stage's winner, which is why this section comes
    before S3 rather than after it.
    """)
    return


@app.cell
def _():
    # The final list, from a selection process that ran outside this notebook.
    # Not re-derived here: only read and displayed. Depth is the layer each
    # configuration was actually scored at, confirmed against the run
    # directories themselves rather than assumed from the stage.
    LEADING_CANDIDATES = [
        {
            "stage": "S1",
            "label": "S1 #1",
            "group": "apertus-8b-instruct_L1-31_rollout_balanced128_hard_bce_lora-none_e1fc6f1f",
            "depth": 15,
            "summary": "rollout_balanced, W=128",
        },
        {
            "stage": "S1",
            "label": "S1 #2",
            "group": "apertus-8b-instruct_L1-31_frontier256_hard_bce_lora-none_2d96b95c",
            "depth": 15,
            "summary": "frontier_window, W=256, centered",
        },
        {
            "stage": "S1",
            "label": "S1 #3",
            "group": "apertus-8b-instruct_L1-31_frontier_hard_negative128_hard_bce_lora-none_710358df",
            "depth": 4,
            "summary": "frontier_window_hard_negative, W=128, centered",
        },
        {
            "stage": "S2a",
            "label": "S2a #1",
            "group": "apertus-8b-instruct_L1-31_frontier512_hard256_bce_lora-none_c1b8cca1",
            "depth": 15,
            "summary": "frontier_window, W=512, centered, horizon=256",
        },
        {
            "stage": "S2a",
            "label": "S2a #2",
            "group": "apertus-8b-instruct_L1-31_all_tokens_hard1024_bce_lora-none_d212fbad",
            "depth": 15,
            "summary": "all_tokens, horizon=1024",
        },
        {
            "stage": "S2a",
            "label": "S2a #3",
            "group": "apertus-8b-instruct_L1-31_frontier512_hard512_bce_lora-none_03e606bc",
            "depth": 15,
            "summary": "frontier_window, W=512, trailing, horizon=512",
        },
        {
            "stage": "S2b",
            "label": "S2b #1",
            "group": "apertus-8b-instruct_L1-31_frontier512_soft_bce_lora-none_ca252971",
            "depth": 15,
            "summary": "soft label, exponential decay/256",
        },
        {
            "stage": "S2b",
            "label": "S2b #2",
            "group": "apertus-8b-instruct_L1-31_frontier512_soft_bce_lora-none_1a539f5c",
            "depth": 15,
            "summary": "soft label, linear decay/256",
        },
        {
            "stage": "S2b",
            "label": "S2b #3",
            "group": "apertus-8b-instruct_L1-31_frontier512_soft_bce_lora-none_53163d18",
            "depth": 15,
            "summary": "soft label, exponential decay/128",
        },
        {
            "stage": "S2d",
            "label": "S2d #1",
            "group": "apertus-8b-instruct_L1-31_frontier128_hard_bce_lora-none_641cb829",
            "depth": 4,
            "summary": "base recipe, pos_weight=on",
        },
        {
            "stage": "S2d",
            "label": "S2d #2",
            "group": "apertus-8b-instruct_L1-31_frontier128_hard_bce_lora-none_17e0fada",
            "depth": 4,
            "summary": "base recipe, pos_weight=off",
        },
        {
            "stage": "S2d",
            "label": "S2d #3",
            "group": "apertus-8b-instruct_L1-31_frontier128_hard_bce_lora-none_465e7640",
            "depth": 4,
            "summary": "base recipe, pos_weight=on, positive_fraction=0.5",
        },
    ]
    STAGE_ORDER = ["S1", "S2a", "S2b", "S2d"]
    return LEADING_CANDIDATES, STAGE_ORDER


@app.cell
def _(LEADING_CANDIDATES, METRIC_ORDER, Path, all_runs, pd):
    # Mirrors load_view()/scorecard_long()'s shape, but keyed by the hardcoded
    # list above rather than by an `exp:` tag: these runs are not tagged for
    # this section and should not be, since run_info.json is written once by
    # the training script and nothing else should rewrite it after the fact.
    _CANDIDATE_VIEWS = [
        ("A", "recall", "view_a_detection", "recall"),
        ("A", "precision", "view_a_detection", "precision"),
        ("B", "in-pattern coverage", "view_b_coverage", "in_pattern_recall"),
        ("B", "warning coverage 128", "view_b_coverage", "warning_recall_128"),
        ("B", "warning coverage 256", "view_b_coverage", "warning_recall_256"),
        ("B", "healthy token rate", "view_b_coverage", "token_false_positive_rate"),
        ("C", "median lead", "view_c_lead_time", "median_offset"),
        ("C", "never fired", "view_c_lead_time", "never_fired_positives"),
    ]
    _CANDIDATE_PERSISTENCE = [
        ("D", "alarm length, degenerate", "positive"),
        ("D", "alarm length, healthy", "negative"),
    ]

    def _candidate_run_dirs(group):
        if all_runs.empty:
            return []
        matches = all_runs[(all_runs["group"] == group) & (all_runs["status"] == "finished")]
        return matches["run_dir"].tolist()

    def _load_candidate_view(run_dirs, depth, view_file):
        frames = []
        for run_dir in run_dirs:
            path = (
                Path(run_dir) / "layers" / f"layer_{depth:02d}" / "evaluation" / "val"
                / f"{view_file}.csv"
            )
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame["run_dir"] = run_dir
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def candidate_scored_seeds():
        """How many of the 36 (12 configs x 3 seeds) already have all four
        views on disk, so a scoring job still in flight is stated rather than
        silently read as zero rows."""
        total = scored = 0
        for candidate in LEADING_CANDIDATES:
            run_dirs = _candidate_run_dirs(candidate["group"])
            total += 3
            for run_dir in run_dirs:
                base = (
                    Path(run_dir) / "layers" / f"layer_{candidate['depth']:02d}"
                    / "evaluation" / "val"
                )
                views = ["view_a_detection", "view_b_coverage", "view_c_lead_time", "view_d_persistence"]
                if all((base / f"{v}.csv").is_file() for v in views):
                    scored += 1
        return scored, total

    def candidate_records(budget):
        """Every view of every leading candidate, one row per seed per metric."""
        records = []
        for candidate in LEADING_CANDIDATES:
            run_dirs = _candidate_run_dirs(candidate["group"])
            if not run_dirs:
                continue
            for view, metric, view_file, column in _CANDIDATE_VIEWS:
                frame = _load_candidate_view(run_dirs, candidate["depth"], view_file)
                if frame.empty or column not in frame.columns:
                    continue
                rows = frame[frame["target_negative_fpr"] == budget]
                for _, row in rows.iterrows():
                    records.append(
                        {
                            "stage": candidate["stage"],
                            "label": candidate["label"],
                            "summary": candidate["summary"],
                            "depth": candidate["depth"],
                            "view": view,
                            "metric": metric,
                            "value": row[column],
                        }
                    )
            persistence = _load_candidate_view(run_dirs, candidate["depth"], "view_d_persistence")
            if persistence.empty:
                continue
            for view, metric, population in _CANDIDATE_PERSISTENCE:
                sub = persistence[persistence["population"] == population]
                rows = sub[sub["target_negative_fpr"] == budget]
                for _, row in rows.iterrows():
                    records.append(
                        {
                            "stage": candidate["stage"],
                            "label": candidate["label"],
                            "summary": candidate["summary"],
                            "depth": candidate["depth"],
                            "view": view,
                            "metric": metric,
                            "value": row["median_first_run_length"],
                        }
                    )
        return pd.DataFrame(records)

    _ = METRIC_ORDER  # re-exported unchanged; the metric set is identical to scorecard()'s
    return candidate_records, candidate_scored_seeds


@app.cell
def _(budget_control):
    candidates_card_budget = budget_control()
    candidates_card_budget
    return (candidates_card_budget,)


@app.cell
def _(
    METRIC_ORDER,
    STAGE_ORDER,
    candidate_records,
    candidate_scored_seeds,
    candidates_card_budget,
    missing,
    mo,
):
    def candidate_scorecard(budget):
        long = candidate_records(budget)
        if long.empty:
            return missing(
                "None of the twelve leading candidates have been through the "
                "evaluator yet on this depth."
            )

        def summarise(values):
            values = values.dropna()
            if values.empty:
                return ""
            middle = values.median()
            text = f"{middle:.3f}" if abs(middle) < 100 else f"{middle:.0f}"
            if len(values) > 1 and values.min() != values.max():
                low, high = values.min(), values.max()
                span = (
                    f"{low:.3f}–{high:.3f}" if abs(middle) < 100 else f"{low:.0f}–{high:.0f}"
                )
                return f"{text} [{span}]"
            return text

        meta = (
            long[["stage", "label", "summary", "depth"]]
            .drop_duplicates()
            .set_index("label")
        )
        pivoted = long.pivot_table(
            index="label", columns="metric", values="value", aggfunc=summarise
        ).reindex(columns=[m for m in METRIC_ORDER if m in set(long["metric"])])
        table = meta.join(pivoted).reset_index()
        stage_rank = {stage: i for i, stage in enumerate(STAGE_ORDER)}
        table["_order"] = table["stage"].map(stage_rank)
        table = table.sort_values(["_order", "label"]).drop(columns="_order")
        table = table.rename(columns={"summary": "recipe", "depth": "scored at layer"})

        scored, total = candidate_scored_seeds()
        progress = (
            f" **{scored} of {total}** seed runs have been scored so far; the "
            "rest are queued or still running, and simply have no row here yet."
            if scored < total
            else ""
        )
        return mo.vstack(
            [
                mo.md(
                    "One row per candidate, grouped by the stage it came from. "
                    "Median across its 3 seeds, with the seed-to-seed range in "
                    "brackets where it varies. Each was scored at the single "
                    "layer it was picked at (**scored at layer**), not the "
                    "automatic per-run checkpoint the rest of the notebook uses "
                    "— these are hand-picked checkpoints, chosen outside this "
                    f"notebook. At a **{budget:.0%} false-alarm budget** on "
                    f"`val`.{progress}"
                ),
                mo.ui.table(table, selection=None),
            ]
        )

    candidate_scorecard(candidates_card_budget.value)
    return


@app.cell
def _(budget_control):
    candidates_views_budget = budget_control()
    candidates_views_budget
    return (candidates_views_budget,)


@app.cell
def _(
    FIGURES,
    INK_MUTED,
    INK_SOFT,
    LEADING_CANDIDATES,
    SERIES,
    STAGE_ORDER,
    candidate_records,
    candidates_views_budget,
    missing,
    mo,
    plt,
    save,
    tidy,
):
    def candidate_views_figure(budget):
        long = candidate_records(budget)
        if long.empty:
            return missing(
                "None of the twelve leading candidates have been through the "
                "evaluator yet on this depth."
            )

        stage_colors = dict(zip(STAGE_ORDER, SERIES[: len(STAGE_ORDER)]))
        stage_of = {c["label"]: c["stage"] for c in LEADING_CANDIDATES}
        ordered_labels = [c["label"] for c in LEADING_CANDIDATES]
        positions = {label: index for index, label in enumerate(ordered_labels)}

        panels = [
            (
                "A. does it fire on the right answers",
                [("recall", "o"), ("precision", "^")],
                None,
            ),
            (
                "B. what share of tokens does it flag",
                [("in-pattern coverage", "o"), ("warning coverage 256", "^")],
                None,
            ),
            ("C. how early, in tokens", [("median lead", "o")], 0.0),
            ("D. how long the first alarm holds", [("alarm length, degenerate", "o")], None),
        ]

        fig, axes = plt.subplots(
            1, len(panels), figsize=(3.5 * len(panels), 0.34 * len(ordered_labels) + 2.8)
        )
        for axis, (title, metrics, reference) in zip(axes, panels):
            for metric, marker in metrics:
                subset = long[long["metric"] == metric]
                if subset.empty:
                    continue
                for label, values in subset.groupby("label")["value"]:
                    y = positions[label]
                    colour = stage_colors[stage_of[label]]
                    if values.min() != values.max():
                        axis.plot(
                            [values.min(), values.max()], [y, y],
                            color=colour, alpha=0.35, linewidth=3, solid_capstyle="round",
                        )
                    axis.plot(
                        values.median(), y, marker, color=colour,
                        markeredgecolor="white", markeredgewidth=0.8,
                        markersize=7 if marker == "o" else 6.5,
                    )
            if reference is not None:
                axis.axvline(reference, color=INK_MUTED, linewidth=1, zorder=0)
            # Boundaries between stages, so the grouping reads without counting rows.
            for boundary in range(3, len(ordered_labels), 3):
                axis.axhline(boundary - 0.5, color=INK_MUTED, alpha=0.4, linewidth=0.8)
            axis.set_yticks(range(len(ordered_labels)))
            axis.set_yticklabels(ordered_labels)
            axis.set_ylim(-0.7, len(ordered_labels) - 0.3)
            axis.invert_yaxis()
            axis.set_title(title, color=INK_SOFT, fontsize=8.5)
            # Which metric is which marker, written under the panel that holds
            # two of them rather than squeezed into its title.
            if len(metrics) > 1:
                shapes = {"o": "circle", "^": "triangle"}
                axis.set_xlabel(
                    "     ".join(
                        f"{shapes.get(marker, marker)} {metric}"
                        for metric, marker in metrics
                    ),
                    fontsize=7.5,
                    color=INK_SOFT,
                )
            tidy(axis)
        for axis in axes[1:]:
            axis.set_yticklabels([])

        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=stage_colors[stage], label=stage)
            for stage in STAGE_ORDER
        ]
        fig.suptitle(
            f"The leading candidates on val, layer noted per row, at a {budget:.0%} "
            "false-alarm budget",
            x=0.005, y=0.99, ha="left", fontsize=10,
        )
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
            ncol=len(STAGE_ORDER), fontsize=8, title="stage", title_fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.93))
        return mo.vstack(
            [
                save(fig, "leading_candidates_views", FIGURES),
                mo.md(
                    "_Same four views as every per-stage plot in this notebook, "
                    "but with all twelve leading candidates on one set of axes, "
                    "coloured by the stage that produced them rather than by "
                    "metric — circles and triangles distinguish the two metrics "
                    "that share a panel, and the dividing lines mark where one "
                    "stage's three picks end and the next stage's begin. Reading "
                    "across a divider: if one stage's cluster sits clearly apart "
                    "from another's on a panel, that stage's axis moved the "
                    "result; if the clusters overlap, this comparison alone "
                    "cannot tell the stages apart on that view._"
                ),
            ]
        )

    candidate_views_figure(candidates_views_budget.value)
    return


@app.cell
def _(LEADING_CANDIDATES, Path, all_runs, mo):
    def candidate_runs(label):
        """The replayed run names behind one candidate, and the depth it was
        scored at."""
        candidate = next(c for c in LEADING_CANDIDATES if c["label"] == label)
        if all_runs.empty:
            return [], candidate["depth"]
        matches = all_runs[
            (all_runs["group"] == candidate["group"])
            & (all_runs["status"] == "finished")
        ]
        names = [Path(d).parent.name for d in matches["run_dir"]]
        return names, candidate["depth"]

    def candidate_control():
        options = [c["label"] for c in LEADING_CANDIDATES]
        return mo.ui.dropdown(options=options, value=options[0], label="candidate")

    return candidate_control, candidate_runs


@app.cell
def _(candidate_control):
    candidates_tradeoff_pick = candidate_control()
    candidates_tradeoff_pick
    return (candidates_tradeoff_pick,)


@app.cell
def _(
    FIGURES,
    INK_MUTED,
    INK_SOFT,
    RAMP,
    SERIES,
    candidate_runs,
    candidates_tradeoff_pick,
    depth_tradeoff_points,
    missing,
    mo,
    mpl,
    plt,
    save,
    tidy,
):
    def candidate_tradeoff_figure(label):
        """Where the depth a candidate was scored at sits among its own depths."""
        runs, depth = candidate_runs(label)
        points = depth_tradeoff_points(runs)
        if points is None:
            return missing(
                "This candidate's runs have no replayed checkpoints, so its other "
                "depths cannot be read."
            )
        grouped = (
            points.groupby("layer")
            .agg(
                warning=("warning", "mean"),
                in_pattern=("in_pattern_recall", "mean"),
            )
            .reset_index()
            .sort_values("layer")
        )

        fig, axis = plt.subplots(figsize=(6.0, 4.0))
        shades = mpl.colors.LinearSegmentedColormap.from_list("depth", RAMP)
        scale = mpl.colors.Normalize(
            vmin=grouped["layer"].min(), vmax=grouped["layer"].max()
        )
        axis.plot(
            grouped["in_pattern"], grouped["warning"],
            color=INK_MUTED, linewidth=0.8, zorder=1,
        )
        axis.scatter(
            grouped["in_pattern"], grouped["warning"],
            c=grouped["layer"], cmap=shades, norm=scale, s=34, zorder=2,
            edgecolors="white", linewidths=0.6,
        )
        if depth in set(grouped["layer"]):
            chosen = grouped[grouped["layer"] == depth].iloc[0]
            axis.plot(
                chosen["in_pattern"], chosen["warning"], "o",
                color=SERIES[1], markersize=12, markerfacecolor="none",
                markeredgewidth=2, zorder=3,
            )
            axis.annotate(
                f"scored at layer {depth}",
                xy=(chosen["in_pattern"], chosen["warning"]),
                xytext=(-8, 8), textcoords="offset points", fontsize=8,
                color=SERIES[1], ha="right",
            )
        axis.margins(0.12)
        axis.set_xlabel("in-pattern coverage")
        axis.set_ylabel("warning coverage, 256")
        axis.set_title(label, color=INK_SOFT)
        bar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=scale, cmap=shades), ax=axis, pad=0.02
        )
        bar.set_label("layer", color=INK_SOFT)
        bar.outline.set_visible(False)
        tidy(axis, xgrid=False)
        fig.suptitle(
            "Was the depth this candidate was scored at the right one?",
            x=0.005, ha="left", fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return mo.vstack(
            [
                save(fig, "candidate_tradeoff", FIGURES),
                mo.md(
                    "_The same trade-off the per-stage figures show, drawn for one "
                    "candidate's own depths and averaged over its three seeds, with "
                    "the depth it was actually scored at ringed. The table above "
                    "reports that one depth; this says what was left on the table "
                    "by choosing it, since a candidate whose ring sits well below "
                    "its own best depth on warning coverage is being judged by a "
                    "pick rather than by its recipe._"
                ),
            ]
        )

    candidate_tradeoff_figure(candidates_tradeoff_pick.value)
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
def _(layer_control, loss_config_control, mo):
    s3_loss_depth = layer_control(value="all", with_all=True)
    s3_loss_config = loss_config_control("S3")
    mo.hstack([s3_loss_config, s3_loss_depth], justify="start", gap=2)
    return s3_loss_config, s3_loss_depth


@app.cell
def _(curve_figure, s3_loss_config, s3_loss_depth):
    curve_figure("S3", s3_loss_depth.value, s3_loss_config.value)
    return


@app.cell
def _(view_controls):
    s3_card_controls = view_controls()
    s3_card_controls
    return (s3_card_controls,)


@app.cell
def _(s3_card_controls, scorecard):
    scorecard(
        "S3",
        s3_card_controls.value["split"],
        s3_card_controls.value["layer"],
        s3_card_controls.value["budget"],
    )
    return


@app.cell
def _(view_controls):
    s3_views_controls = view_controls()
    s3_views_controls
    return (s3_views_controls,)


@app.cell
def _(s3_views_controls, views_figure):
    views_figure(
        "S3",
        s3_views_controls.value["split"],
        s3_views_controls.value["layer"],
        s3_views_controls.value["budget"],
    )
    return


@app.cell
def _(config_control):
    s3_tradeoff_config = config_control("S3")
    s3_tradeoff_config
    return (s3_tradeoff_config,)


@app.cell
def _(s3_tradeoff_config, stage_tradeoff_figure):
    stage_tradeoff_figure("S3", s3_tradeoff_config.value)
    return


@app.cell
def _(config_control, layer_control, mo):
    s3_steps_config = config_control("S3")
    s3_steps_depth = layer_control()
    mo.hstack([s3_steps_config, s3_steps_depth], justify="start", gap=2)
    return s3_steps_config, s3_steps_depth


@app.cell
def _(s3_steps_config, s3_steps_depth, stage_tradeoff_trajectory):
    stage_tradeoff_trajectory(
        "S3", s3_steps_config.value, s3_steps_depth.value
    )
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

    The pickers below offer only the two test splits. Validation is deliberately
    unreachable here: every threshold was frozen on it and every recipe chosen
    against it, so reporting it in this stage would answer a different question
    than the one the stage exists to ask.
    """)
    return


@app.cell
def _(test_view_controls):
    s4_card_controls = test_view_controls()
    s4_card_controls
    return (s4_card_controls,)


@app.cell
def _(s4_card_controls, scorecard):
    scorecard(
        "S4",
        s4_card_controls.value["split"],
        s4_card_controls.value["layer"],
        s4_card_controls.value["budget"],
    )
    return


@app.cell
def _(test_view_controls):
    s4_views_controls = test_view_controls()
    s4_views_controls
    return (s4_views_controls,)


@app.cell
def _(s4_views_controls, views_figure):
    views_figure(
        "S4",
        s4_views_controls.value["split"],
        s4_views_controls.value["layer"],
        s4_views_controls.value["budget"],
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Part 3. Does a probe transfer to another model?

    Every run above trains and scores against Apertus's own activations. This
    part asks a different question of the same checkpoints: does a head learned
    on Apertus still separate degenerating answers on a model it has never
    seen, with no retraining at all?

    A checkpoint is scored against a different model's own dataset build, using
    the same per-token scoring and the same four-view evaluator as scoring it on
    Apertus, just pointed at cached activations from
    `meta-llama/Llama-3.1-8B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.1`.
    Both share Apertus's hidden size of 4096 and its 32 layers, so a saved
    linear head is shape-compatible. Whether it is also *useful* there is what
    this part reports.

    The twelve leading candidates are what gets transferred, three seeds each,
    every one at the depth it was scored at. Only frozen checkpoints are ever
    scored this way: a LoRA adapter is fitted to Apertus's own decoder weights
    and has no meaning against another model's.

    Everything here reads the validation split. The transferred side has the two
    test splits on disk as well, but the Apertus-side numbers each result is
    compared against exist only on validation, so a comparison on anything else
    would have nothing to sit beside.

    Ground truth for the target models comes from the same LLM judge as every
    other split in this notebook, so a row below is only as trustworthy as that
    judge run on that model's answers.

    Two further target models, both Apertus 1.5 variants, are being scored the
    same way. That sweep covers five of the thirty-six seed runs so far, so it
    is left out of this part rather than shown as a ragged column.
    """)
    return


@app.cell
def _(LEADING_CANDIDATES, Path, all_runs, pd):
    # The two models every candidate has been scored against. Discovery is by
    # candidate rather than by walking the output tree, so a half-finished sweep
    # against some other model cannot quietly add itself to the comparison.
    CROSS_MODEL_TARGETS = {
        "llama3p1-8b-instruct": "llama 3.1 8b",
        "mistral-7b-instruct-v0p1": "mistral 7b",
    }
    CROSS_MODEL_SPLIT = "val"

    def cross_model_index():
        """One row per (candidate, seed, target model) that has been scored."""
        if all_runs.empty:
            return pd.DataFrame()
        finished = all_runs[all_runs["status"] == "finished"]
        rows = []
        for candidate in LEADING_CANDIDATES:
            matches = finished[finished["group"] == candidate["group"]]
            for run in matches.itertuples():
                depth = candidate["depth"]
                root = Path(run.run_dir)
                native = root / "layers" / f"layer_{depth:02d}"
                for target in CROSS_MODEL_TARGETS:
                    scoped = root / "cross_model" / target / f"layer_{depth:02d}"
                    evaluation = scoped / "evaluation" / CROSS_MODEL_SPLIT
                    if not (evaluation / "view_a_detection.csv").is_file():
                        continue
                    rows.append(
                        {
                            "stage": candidate["stage"],
                            "label": candidate["label"],
                            "recipe": candidate["summary"],
                            "depth": depth,
                            "target": CROSS_MODEL_TARGETS[target],
                            "seed": run.seed,
                            "evaluation": str(evaluation),
                            "scores": str(scoped / "scores" / f"{CROSS_MODEL_SPLIT}.parquet"),
                            "native_scores": str(
                                native / "scores" / f"{CROSS_MODEL_SPLIT}.parquet"
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    CROSS_MODEL = cross_model_index()
    return (CROSS_MODEL,)


@app.cell
def _(CROSS_MODEL, LEADING_CANDIDATES, missing, mo):
    mo.stop(
        CROSS_MODEL.empty,
        missing(
            "No cross-model scores on disk yet. They land under "
            "`<run_dir>/cross_model/<model>/layer_NN/evaluation/<split>/` once "
            "`scripts/evaluate_cross_model_transfer.py` and "
            "`scripts/evaluate_scores.py` have run for a checkpoint."
        ),
    )
    mo.md(
        f"**{len(CROSS_MODEL)} scored transfers found**, out of "
        f"{len(LEADING_CANDIDATES) * 3 * 2} expected: twelve candidates, three "
        "seeds each, two target models."
    )
    return


@app.cell
def _(budget_control):
    transfer_table_budget = budget_control()
    transfer_table_budget
    return (transfer_table_budget,)


@app.cell
def _(CROSS_MODEL, Path, mo, pd, transfer_table_budget):
    # The scorecard's own metric set, plus the two rollout-level rates this part
    # states outright. TPR and recall are the same quantity, which is worth
    # showing rather than asserting.
    _TRANSFER_VIEWS = [
        ("recall", "view_a_detection", "recall"),
        ("precision", "view_a_detection", "precision"),
        ("TPR", "view_a_detection", "recall"),
        ("FPR", "view_a_detection", "negative_fpr"),
        ("in-pattern coverage", "view_b_coverage", "in_pattern_recall"),
        ("warning coverage 128", "view_b_coverage", "warning_recall_128"),
        ("warning coverage 256", "view_b_coverage", "warning_recall_256"),
        ("healthy token rate", "view_b_coverage", "token_false_positive_rate"),
        ("median lead", "view_c_lead_time", "median_offset"),
        ("never fired", "view_c_lead_time", "never_fired_positives"),
    ]
    _TRANSFER_PERSISTENCE = [
        ("alarm length, degenerate", "positive"),
        ("alarm length, healthy", "negative"),
    ]
    _TRANSFER_ORDER = [name for name, *_ in _TRANSFER_VIEWS] + [
        name for name, _ in _TRANSFER_PERSISTENCE
    ]

    def transfer_records(budget):
        """Every view of every transfer, one row per seed per metric."""
        records = []
        for record in CROSS_MODEL.itertuples():
            evaluation = Path(record.evaluation)
            for metric, view_file, column in _TRANSFER_VIEWS:
                path = evaluation / f"{view_file}.csv"
                if not path.is_file():
                    continue
                frame = pd.read_csv(path)
                if column not in frame.columns:
                    continue
                rows = frame[frame["target_negative_fpr"] == budget]
                for _, row in rows.iterrows():
                    records.append(
                        {
                            "stage": record.stage,
                            "label": record.label,
                            "recipe": record.recipe,
                            "target": record.target,
                            "metric": metric,
                            "value": row[column],
                        }
                    )
            path = evaluation / "view_d_persistence.csv"
            if not path.is_file():
                continue
            persistence = pd.read_csv(path)
            for metric, population in _TRANSFER_PERSISTENCE:
                rows = persistence[
                    (persistence["population"] == population)
                    & (persistence["target_negative_fpr"] == budget)
                ]
                for _, row in rows.iterrows():
                    records.append(
                        {
                            "stage": record.stage,
                            "label": record.label,
                            "recipe": record.recipe,
                            "target": record.target,
                            "metric": metric,
                            "value": row["median_first_run_length"],
                        }
                    )
        return pd.DataFrame(records)

    def transfer_table(budget):
        long = transfer_records(budget)
        if long.empty:
            return mo.md("_Nothing scored at this budget yet._")

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

        meta = (
            long[["stage", "label", "recipe", "target"]]
            .drop_duplicates()
            .set_index(["label", "target"])
        )
        pivoted = long.pivot_table(
            index=["label", "target"],
            columns="metric",
            values="value",
            aggfunc=summarise,
        ).reindex(columns=[m for m in _TRANSFER_ORDER if m in set(long["metric"])])
        table = meta.join(pivoted).reset_index()
        table = table.sort_values(["label", "target"])
        return mo.vstack(
            [
                mo.md(
                    "One row per candidate per target model: the median across "
                    "its three seeds, with the seed-to-seed range in brackets "
                    f"where it varies, at a **{budget:.0%} false-alarm budget** "
                    "on the target model's own validation split. **TPR** is the "
                    "same quantity as **recall**, reported under both names "
                    "because the pair with **FPR** is how a detection rate is "
                    "usually read; **FPR** is the realised rollout-level "
                    "false-alarm rate, which is the budget as actually spent "
                    "rather than as requested."
                ),
                mo.ui.table(table, selection=None),
            ]
        )

    transfer_table(transfer_table_budget.value)
    return


@app.cell
def _(CROSS_MODEL, np, pd):
    # The stored views hold three budgets. These plots need the whole axis, so
    # thresholds are recomputed from the per-token scores at a fine grid, using
    # the same function the evaluator itself calls at those three points.
    from degeneration_probe.evaluation.protocol import (
        coverage_window,
        rollout_score,
        threshold_for_budget,
    )

    BUDGET_GRID = np.geomspace(0.001, 0.30, 48)
    _SWEEP_CACHE = {}

    def _sweep_one(path):
        """Recall, precision and in-pattern coverage across the whole budget axis.

        One pass over the file answers every budget: a rollout fires at tau
        exactly when its highest score reaches tau, and in-pattern coverage at
        tau is a position in the sorted pool of in-pattern token scores.
        """
        if path in _SWEEP_CACHE:
            return _SWEEP_CACHE[path]
        frame = pd.read_parquet(path)
        positives = frame[frame["is_positive"].astype(bool)]
        negatives = frame[~frame["is_positive"].astype(bool)]
        if positives.empty or negatives.empty:
            _SWEEP_CACHE[path] = pd.DataFrame()
            return _SWEEP_CACHE[path]
        negative_peaks = np.array(
            [rollout_score(s) for s in negatives["scores"]], dtype=np.float64
        )
        positive_peaks = np.array(
            [rollout_score(s) for s in positives["scores"]], dtype=np.float64
        )
        in_pattern = np.sort(
            np.concatenate(
                [
                    np.asarray(scores, dtype=np.float64)[
                        coverage_window(len(scores), int(onset), None)
                    ]
                    for scores, onset in zip(
                        positives["scores"], positives["onset_position"]
                    )
                ]
            )
        )
        rows = []
        for budget in BUDGET_GRID:
            tau, _ = threshold_for_budget(negative_peaks, float(budget))
            caught = int((positive_peaks >= tau).sum())
            false_alarms = int((negative_peaks >= tau).sum())
            flagged = caught + false_alarms
            rows.append(
                {
                    "budget": float(budget),
                    "recall": caught / positive_peaks.size,
                    "precision": (caught / flagged) if flagged else 0.0,
                    "in_pattern_recall": float(
                        (in_pattern.size - np.searchsorted(in_pattern, tau, "left"))
                        / in_pattern.size
                    ),
                }
            )
        _SWEEP_CACHE[path] = pd.DataFrame(rows)
        return _SWEEP_CACHE[path]

    def transfer_sweeps():
        """The budget axis for every transfer, and for its native counterpart."""
        frames = []
        for record in CROSS_MODEL.itertuples():
            for source, path in (
                (record.target, record.scores),
                ("apertus (native)", record.native_scores),
            ):
                curve = _sweep_one(path)
                if curve.empty:
                    continue
                piece = curve.copy()
                piece["label"] = record.label
                piece["source"] = source
                piece["seed"] = record.seed
                frames.append(piece)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    SWEEPS = transfer_sweeps()
    return (SWEEPS,)


@app.cell
def _(
    FIGURES,
    INK_SOFT,
    LEADING_CANDIDATES,
    SERIES,
    SWEEPS,
    missing,
    mo,
    plt,
    save,
    tidy,
):
    def budget_sweep_figure(column, title, ylabel, name):
        """One metric against the false-alarm budget, per candidate.

        A panel per candidate rather than every line on one axes: twelve
        candidates times two target models plus twelve native references is
        thirty-six lines, and the comparison that matters is within a candidate,
        not across them.
        """
        if SWEEPS.empty:
            return missing("No transfer has been swept yet.")
        labels = [c["label"] for c in LEADING_CANDIDATES if c["label"] in set(SWEEPS["label"])]
        if not labels:
            return missing("No transfer has been swept yet.")
        sources = sorted(s for s in SWEEPS["source"].unique() if s != "apertus (native)")
        colours = dict(zip(sources, SERIES))

        columns = 4
        rows = -(-len(labels) // columns)
        fig, axes = plt.subplots(
            rows, columns, figsize=(3.0 * columns, 2.5 * rows), squeeze=False,
            sharex=True, sharey=True,
        )
        flat = [axis for row in axes for axis in row]
        for axis, label in zip(flat, labels):
            panel = SWEEPS[SWEEPS["label"] == label]
            for source in sources:
                line = panel[panel["source"] == source].groupby("budget")[column].median()
                if line.empty:
                    continue
                axis.plot(line.index, line.to_numpy(), color=colours[source], label=source)
            native = (
                panel[panel["source"] == "apertus (native)"]
                .groupby("budget")[column]
                .median()
            )
            if not native.empty:
                axis.plot(
                    native.index,
                    native.to_numpy(),
                    color=INK_SOFT,
                    linestyle="--",
                    linewidth=1.4,
                    label="apertus (native)",
                )
            axis.set_xscale("log")
            axis.set_ylim(0, 1)
            axis.set_title(label, color=INK_SOFT)
            tidy(axis, xgrid=False)
        for axis in flat[len(labels):]:
            axis.set_axis_off()
        for axis in axes[-1]:
            axis.set_xlabel("false-alarm budget")
        for row in axes:
            row[0].set_ylabel(ylabel)
        handles, names = flat[0].get_legend_handles_labels()
        fig.legend(
            handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.02),
            ncol=len(names),
        )
        fig.suptitle(title, x=0.005, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        return mo.vstack(
            [
                save(fig, name, FIGURES),
                mo.md(
                    "_One panel per candidate, median across its three seeds. "
                    "Solid lines are the head scored on a model it never saw; the "
                    "dashed line is the same head on the Apertus validation split "
                    "it was trained and picked on, read at the same budget. The "
                    "gap between them is what transfer costs. The budget axis is "
                    "logarithmic because the operating points that matter are "
                    "small, and it is swept continuously rather than at the three "
                    "stored points, which needs the thresholds recomputed from the "
                    "per-token scores._"
                ),
            ]
        )

    return (budget_sweep_figure,)


@app.cell
def _(budget_sweep_figure):
    budget_sweep_figure(
        "recall",
        "Detection: does it still catch a degenerate answer on another model?",
        "recall",
        "transfer_recall_vs_budget",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    **In-pattern coverage below is genuinely zero, and it is not a plotting
    fault.** It is worth saying why, because zero everywhere usually means a
    broken axis.

    Transferred to another model, the head stops separating anything: on Llama
    every healthy answer peaks somewhere between 0.58 and 0.97, and the tokens
    inside the loop reach at most 0.94. The threshold that spends even a 1%
    false-alarm budget therefore sits at about 0.95, above the highest score any
    in-loop token achieves, so nothing inside the loop is flagged and the rate is
    exactly 0. On Apertus the same head puts in-loop tokens near 0.93 on average
    while healthy answers spread all the way down to 0.14, and the same 1%
    threshold still admits two thirds of them.

    So the head does not produce a weaker version of the same signal on another
    model. It produces a compressed, uniformly high score that carries almost no
    ordering, which is also why rollout recall sits near zero at small budgets.
    """)
    return


@app.cell
def _(budget_sweep_figure):
    budget_sweep_figure(
        "in_pattern_recall",
        "Coverage: does it still flag the tokens inside the loop?",
        "in-pattern coverage",
        "transfer_in_pattern_vs_budget",
    )
    return


@app.cell
def _(budget_sweep_figure):
    budget_sweep_figure(
        "precision",
        "Precision: what share of its alarms are on degenerate answers?",
        "precision",
        "transfer_precision_vs_budget",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Appendix

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

    # Score one depth of a finished run against another model's dataset build.
    python scripts/evaluate_cross_model_transfer.py --run-dir outputs/<run>/latest --layer 15 --dataset-config configs/dataset/degeneration-dataset-llama3p1-8b-instruct.yaml --splits val
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
      cross_model/<model>/layer_NN/     the same depth, scored on another model
      checkpoint_replay.parquet         every saved checkpoint, every depth
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
