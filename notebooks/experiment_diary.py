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

    **fired_before_frontier**, the share of the answers that fired whose alarm
    landed before the loop, is the third number of the view and the one that
    makes the other two readable. The offsets are not a single population with
    a centre: an answer is either caught in the run-up, typically a couple of
    hundred tokens ahead, or it is caught shortly after the loop has started,
    and almost nothing sits in between. A median over both halves reports a
    point where few answers actually are, so it should never be read without
    the share it was taken over.

    #### Warning coverage and median lead describe different answers

    Read side by side these two look inconsistent: warning coverage says a
    fair number of tokens before the frontier get flagged, while the median
    lead says the typical alarm arrives after it. They are not in conflict,
    and the reason is worth stating once because it is easy to lose.

    An alarm is the *first* flagged token. So if any token inside the warning
    band is flagged, the alarm is at or before that token, hence before the
    frontier. The converse is what matters: an answer whose alarm lands after
    the frontier has, by construction, nothing flagged in its band and
    contributes exactly zero to warning coverage. Warning coverage is
    therefore produced entirely by the answers that fire early, while the
    median lead is taken over every answer that fired at all.

    The two numbers are related by the share above. Pooled warning coverage is
    roughly `fired_before_frontier` times the coverage those early answers
    achieve inside their own bands, and that second factor is several times
    the pooled figure. A pooled warning coverage of 0.20 is not a probe that
    flags a fifth of every run-up; it is a probe that flags close to half the
    run-up on the two fifths of answers it sees coming, and none of it on the
    rest. Reporting the three numbers together, coverage, share fired early,
    and median lead, is what makes that shape visible.

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
    ### How a run is trained

    Every run in the programme is trained the same way, so that a difference
    between two of them is a difference in the recipe rather than in the effort
    spent on it.

    - **A step sees 4,096 supervised tokens.** An epoch is not a fixed amount of
      learning here: under the exhaustive rule it is the whole corpus, under an
      anchored-window rule it is one window per answer. Training each recipe for
      one epoch would hand them budgets differing by an order of magnitude, and
      the measured difference would be mostly a difference in training length.
      The number of answers a step reads is whatever it takes to reach that
      token budget, measured from the training stream rather than assumed from
      the window size.
    - **The probe is measured and saved every 50 steps**, and every checkpoint
      is kept. A checkpoint holds a linear head and, where adapters are trained,
      the adapters: a few megabytes against a frozen backbone the configuration
      already describes in full. Keeping all of them is what allows the rule
      below to be reconsidered without retraining anything.
    - **A run stops when the rule says it has stopped improving.** A step cap of
      2000 stands behind that as a backstop and is not the operating point: over
      the runs measured so far, 99% of reported checkpoints were reached by step
      1500 and the median by step 800.

    Two differences between the frozen and the adapted stages are forced rather
    than chosen. A frozen run trains a head at all thirty-one depths in one pass,
    because reading one depth out of the cache costs almost what reading every
    depth costs; an adapted run hooks a single layer, so it names the depth its
    frozen counterpart was best at. And an adapted run reads one answer at a
    time because the language model is resident, reaching the same token budget
    by accumulation instead. Holding the token budget therefore costs a narrow
    window far more with adapters than without, since supervising sixty-four
    tokens still runs a whole answer through the model.

    ### Which checkpoint gets reported

    A run keeps forty or more checkpoints at every depth it trains. Which one a
    depth is judged on is decided by the rule below, applied to each depth
    independently, since the heads share no parameters and plateau at very
    different steps. It replaces an earlier rule that watched rollout-level
    recall, which saturates: on a split holding a hundred degenerate answers that
    number reads the same at every step of a run, so the checkpoint it names is
    whichever one happened to catch one more answer.

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

    ### Where the four numbers come from

    $W = 256$, $g = 0.3$, $\epsilon = 0.002$, $P = 4$. Every checkpoint of every
    run was kept, so each combination can be read off the recorded trajectories
    rather than argued about, and what that reading says is that **these four
    numbers set how long a run trains, not what it reports**. Swept over floors
    from 0 to 0.5, tolerances from 0.0005 to 0.004 and patiences from 2 to 8, the
    depth a run is reported at does not move at all and the value it reports
    moves by a few percent, while the step at which it stops roughly doubles.
    They are a budget, and should be read as one.

    The two that could be argued from the data are $\epsilon$ and $P$.

    - **$\epsilon$** is one sampling error of the objective. Splitting the
      validation answers in half and measuring each half separately puts that
      error at 0.0017 against a typical peak of 0.022, and puts the movement
      between neighbouring checkpoints at 0.0004. So 0.002 is about one standard
      error and about five times the step-to-step wobble: small enough to notice
      real improvement, large enough not to chase noise.
    - **$P$** was chosen by asking what more of it buys. Fitting the rule on one
      half of the answers and re-measuring its pick on the other, raising
      patience from 2 to 8 lifts the value claimed on the half it selected on by
      12% and the honest value on the held-out half by 2.6%, for double the
      training. Most of what waiting longer appears to buy is the selection
      finding a higher point on the same noise. At $P = 4$ the rule already
      captures 88% of what an oracle with hindsight would have picked.

    $g$ is a judgement rather than a result. Coverage inside the loop has no
    natural break: across every depth of every run it is a smooth distribution,
    with about a tenth of them sitting within 0.05 of the floor wherever the
    floor is put. What 0.3 does is separate the depths that never found the loop
    from the rest, and moving it does not change which depth any run reports.

    All of this is small next to the seed. Repeats of one recipe differ by 18% in
    the value they report, and by three layers in the depth. **Nothing about the
    rule's settings is worth as much as another seed.**

    ### What the rule is for

    **Only the checkpoint the rule picks is ever scored in full.** A *full
    scoring pass* means writing one score for every token of all 3,634
    validation answers and putting them through all four views above. It takes
    a few minutes for one depth at one checkpoint. A frozen run has 31 depths and
    40 checkpoints, so doing it everywhere would cost days per run and throw away
    all but one row of the result. The rule exists so that cost is spent once,
    at the checkpoint worth spending it on.

    **The rule is one function, used at either moment.** It reads a sequence of
    cheap per-checkpoint measurements and says where a depth stopped improving
    and which checkpoint was its best. That sequence can arrive as training
    produces it, in which case the rule ends the run, or it can be read back
    afterwards from saved checkpoints, in which case it picks which one to
    report. Both call the same code, so the two cannot disagree.

    Which of the two a stage uses is settled by cost rather than preference. A
    frozen probe is a linear map on states that do not change, so a whole run's
    checkpoints collapse into one matrix and every one of them can be applied to
    answers that are read once: replaying a finished run costs about one pass
    over the data rather than one pass per checkpoint, which is why the frozen
    stages are judged that way. Adapters change the states, so nothing collapses
    and there is no cheap replay; an adapted run therefore measures the rule as
    it trains, which costs nothing beyond the evaluation it was already doing.

    One thing the two do not share is how firmly the threshold is pinned. The
    objective is measured at a fixed false-alarm budget of one percent, so the
    threshold sits above the highest-scoring one percent of healthy answers, and
    how many healthy answers there are decides how much it moves. Against the
    whole validation split it rests on the top thirty-five; against a monitor of
    four hundred answers it rests on three, and the objective derived from it
    then carries a relative spread of 46%. A replayed run reads the whole split
    by construction. A run judged live has to monitor the whole split as well,
    and does.
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

    # A frozen run's trajectory is recovered afterwards by applying every saved
    # checkpoint to answers read once; an adapted run records the same numbers as
    # it trains, because adapters leave nothing to collapse and so nothing to
    # replay cheaply. The two files hold the same columns, so everything
    # downstream reads whichever a run left behind.
    TRAJECTORY_FILES = ("checkpoint_replay.parquet", "selection_history.parquet")

    def trajectory_path(run_dir):
        for name in TRAJECTORY_FILES:
            path = Path(run_dir) / name
            if path.is_file():
                return path
        return None

    def replayed_runs():
        """Runs whose checkpoints have been put through the rule."""
        if all_runs.empty:
            return {}
        found = {}
        for row in all_runs.itertuples():
            path = trajectory_path(row.run_dir)
            if path is not None:
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
                    "run: **never selectable** is how many of its depths never "
                    "cleared the in-loop-coverage floor at all, "
                    "**earliest/median stop** is when its depths' patience ran out, "
                    "**best depth** is the one the run would actually be reported "
                    "at. A run ends when its **slowest** depth stops, so that "
                    "column is what a future run's walltime has to cover."
                ),
                mo.ui.table(pd.DataFrame(rows), selection=None),
            ]
        )

    return stopping_outcomes, trajectory_path


@app.cell
def _(stopping_outcomes):
    stopping_outcomes()
    return


@app.cell
def _(Path, all_runs, pd, trajectory_path):
    # The quantity selection runs on, so the quantity whose shape decides whether
    # a longer run would have changed the answer.
    SELECTION_METRIC = "warning_recall_256"

    def replay_history():
        """Every replayed checkpoint of every run, carrying the run's settings."""
        if all_runs.empty:
            return pd.DataFrame()
        frames = []
        for row in all_runs.itertuples():
            path = trajectory_path(row.run_dir)
            if path is None:
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
    mo.md(r"""
    ### What selecting on an answer-level metric would cost

    The rule selects on coverage of the run-up. The alternative is to select on
    the quantity this literature reports, and the cost of doing so is measurable
    without training anything, because the replay already holds a realised recall
    and a realised false-alarm rate at every depth and every saved step. With the
    split's populations, 108 degenerate answers and 3,532 negatives, those two
    give the whole confusion matrix, so accuracy and F1 come out of the replay
    arithmetic. AUC does not: it needs the score distribution rather than one
    operating point, and per-token scores exist at only one checkpoint per run.

    Eighty-eight frozen multi-layer runs, candidate set held identical at every
    saved step of every depth, so only the objective changes:

    | objective | warning coverage | times worse | alarm later | in-pattern | accuracy |
    |---|---|---|---|---|---|
    | warning coverage 256 (reference) | 0.0372 | 1.0 | 0 | 0.707 | 0.9901 |
    | accuracy | 0.0183 | **2.4** | 10 | 0.619 | 0.9904 |
    | answer recall | 0.0115 | **3.4** | 25 | 0.509 | 0.9901 |
    | F1 | 0.0213 | 1.9 | 8 | 0.648 | 0.9904 |
    | the deployed rule | 0.0336 | 1.1 | 0 | 0.696 | 0.9901 |

    All medians over the 88 runs. The last row is the rule as it actually runs,
    with its eligibility floor and its patience, against an unconstrained argmax
    of its own objective: it gives up 1.1x, which is the price of stopping early
    and is worth knowing separately from the price of the objective.

    **What the trade actually buys.** Accuracy at the checkpoint chosen for
    warning coverage is 0.9901. Accuracy at the checkpoint chosen for accuracy is
    0.9904. So selecting on accuracy gives up a factor of 2.4 in coverage of the
    run-up to gain **0.0003** in accuracy.

    Two statistics say why, and they are the sharpest form of the argument:

    - Within a single run, **1,179 of 1,240** candidate checkpoint-and-depth
      combinations sit within one accuracy point of the best. Ninety-five percent
      of the grid is indistinguishable.
    - Over the same grid, warning coverage spans a median fold range of
      **692x**.

    At the reported depth alone, over the 40 saved checkpoints of one run, answer
    recall runs from 0.9907 to 1.0000 while warning coverage runs from 0.0042 to
    0.0437, a factor of ten.

    An AUC row is pending: it needs a checkpoint-by-checkpoint scoring pass, which
    is running at the reported depth.

    ```bash
    # every saved checkpoint of one depth, each into its own directory with its
    # own provenance, so the checkpoint behind each number can be named
    sbatch --time=08:00:00 cluster/score_selected_depths.sbatch \
      outputs/<run>/latest "15:50 15:100 15:150 ... 15:2000" val
    ```
    """)
    return


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
        DEFAULT_LAYER,
        LADDER_RUNGS,
        POSITIONAL_RUNGS,
        PROBED_LAYERS,
        SEEDS,
        WINDOWS,
    )


@app.cell
def _(BASE, LADDER_RUNGS, POSITIONAL_RUNGS, SEEDS, WINDOWS):
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
        # The recipes each earlier stage selected: its leader, and everything within
        # one standard deviation of it, since at that separation the ordering between
        # them is a property of the three seeds rather than of the recipes. Each is
        # carried at the depth its frozen counterpart was best at, averaged over
        # seeds: a single seed's best depth moves by ten layers between repeats,
        # while the averaged profile does not. The walltime each asks for follows
        # from its window: holding the token budget makes a narrow window run many
        # more answers through the model per step than a wide one.
        CARRIED = [
        # S1: rollout_balanced W=128/centered hard h=0 bce (0.0423 frozen)
        {
            "stage": "S1",
            "depth": 15,
            "walltime": "12:00:00",
            "overrides": {
                "training.selection.strategy": "rollout_balanced",
                "training.selection.window_size": 128,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_hard",
                "training.label.horizon": 0,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        # S1: frontier_window W=256/centered hard h=0 bce (0.0370 frozen)
        {
            "stage": "S1",
            "depth": 15,
            "walltime": "09:00:00",
            "overrides": {
                "training.selection.strategy": "frontier_window",
                "training.selection.window_size": 256,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_hard",
                "training.label.horizon": 0,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        # S1: frontier_window_hard_negative W=256/centered hard h=0 bce (0.0365 frozen)
        {
            "stage": "S1",
            "depth": 15,
            "walltime": "08:00:00",
            "overrides": {
                "training.selection.strategy": "frontier_window_hard_negative",
                "training.selection.window_size": 256,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_hard",
                "training.label.horizon": 0,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        # S1: random_window W=64/centered hard h=0 bce (0.0357 frozen)
        {
            "stage": "S1",
            "depth": 12,
            "walltime": "12:00:00",
            "overrides": {
                "training.selection.strategy": "random_window",
                "training.selection.window_size": 64,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_hard",
                "training.label.horizon": 0,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        # S1: all_tokens W=128 hard h=0 bce (0.0355 frozen)
        {
            "stage": "S1",
            "depth": 12,
            "walltime": "12:00:00",
            "overrides": {
                "training.selection.strategy": "all_tokens",
                "training.selection.window_size": 128,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_hard",
                "training.label.horizon": 0,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        # S2a: frontier_window W=512/centered hard h=256 bce (0.0618 frozen)
        {
            "stage": "S2a",
            "depth": 15,
            "walltime": "07:00:00",
            "overrides": {
                "training.selection.strategy": "frontier_window",
                "training.selection.window_size": 512,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_hard",
                "training.label.horizon": 256,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        # S2a: all_tokens W=128 hard h=1024 bce (0.0617 frozen)
        {
            "stage": "S2a",
            "depth": 15,
            "walltime": "12:00:00",
            "overrides": {
                "training.selection.strategy": "all_tokens",
                "training.selection.window_size": 128,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_hard",
                "training.label.horizon": 1024,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        # S2a: frontier_window W=512/trailing hard h=512 bce (0.0592 frozen)
        {
            "stage": "S2a",
            "depth": 15,
            "walltime": "07:00:00",
            "overrides": {
                "training.selection.strategy": "frontier_window",
                "training.selection.window_size": 512,
                "training.selection.anchor": "trailing",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_hard",
                "training.label.horizon": 512,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        # S2a: all_tokens W=128 hard h=256 bce (0.0535 frozen)
        {
            "stage": "S2a",
            "depth": 15,
            "walltime": "12:00:00",
            "overrides": {
                "training.selection.strategy": "all_tokens",
                "training.selection.window_size": 128,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_hard",
                "training.label.horizon": 256,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        # S2b: frontier_window W=512/centered soft exponential 256 bce posw=off (0.0518 frozen)
        {
            "stage": "S2b",
            "depth": 15,
            "walltime": "07:00:00",
            "overrides": {
                "training.selection.strategy": "frontier_window",
                "training.selection.window_size": 512,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_soft",
                "training.label.horizon": 0,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "false",
            },
        },
        # S2b: frontier_window W=512/centered soft linear 256 bce posw=off (0.0456 frozen)
        {
            "stage": "S2b",
            "depth": 15,
            "walltime": "07:00:00",
            "overrides": {
                "training.selection.strategy": "frontier_window",
                "training.selection.window_size": 512,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_soft",
                "training.label.horizon": 0,
                "training.label.decay": "linear",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "false",
            },
        },
        # S2c: frontier_window W=128/centered repetition_score mse (0.0038 frozen)
        {
            "stage": "S2c",
            "depth": 11,
            "walltime": "12:00:00",
            "overrides": {
                "training.selection.strategy": "frontier_window",
                "training.selection.window_size": 128,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "token_signal",
                "training.label.horizon": 0,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "mse",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        # S2d: frontier_window W=128/centered hard h=0 bce (0.0321 frozen)
        {
            "stage": "S2d",
            "depth": 4,
            "walltime": "12:00:00",
            "overrides": {
                "training.selection.strategy": "frontier_window",
                "training.selection.window_size": 128,
                "training.selection.anchor": "centered",
                "training.selection.positive_fraction": 0.25,
                "training.label.family": "frontier_hard",
                "training.label.horizon": 0,
                "training.label.decay": "exponential",
                "training.label.decay_length": 256,
                "training.label.signal": "repetition_score",
                "training.loss.name": "bce",
                "training.loss.bce.use_pos_weight": "true",
            },
        },
        ]
        runs = []
        for entry in CARRIED:
            for seed in SEEDS:
                runs.append(
                    {
                        "label": (
                            f"{entry['stage']} leader, adapted at depth "
                            f"{entry['depth']} (seed {seed})"
                        ),
                        "overrides": recipe(
                            **entry["overrides"],
                            **{
                                "training.features.regime": "adapted",
                                "training.lora.enabled": "true",
                                "training.lora.layers": "all",
                                "training.lora.rank": 16,
                                # The head starts as noise, so adapters that
                                # moved at the head's rate would rewrite the
                                # representation before the probe knew what it
                                # was looking for.
                                "training.optimizer.lora_learning_rate": 1e-5,
                                # One depth rather than every depth: the probe
                                # reads its layer through a hook on a resident
                                # model, so a head per depth is not available
                                # here the way it is over cached states.
                                "training.probe.layers": "null",
                                "training.probe.layer": entry["depth"],
                                # One answer at a time, for the same reason. The
                                # token budget is unchanged and reached by
                                # accumulation instead.
                                "training.runtime.per_device_train_batch_size": 1,
                                # The threshold sits above the highest-scoring
                                # one percent of healthy answers, so a monitor
                                # of four hundred would rest it on three of
                                # them. There is no cheap replay to fall back
                                # on once adapters have moved the states, so the
                                # whole split is read as the run trains.
                                "training.validation.max_rollouts": "null",
                                "training.stopping.enabled": "true",
                                "training.stopping.floor": 0.3,
                                "training.stopping.band": 256,
                                "training.stopping.tolerance": 0.002,
                                "training.stopping.patience": 4,
                                "training.runtime.seed": seed,
                            },
                        ),
                        "sbatch": f"--time={entry['walltime']}",
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
            "shell": [
                # The four scorers the labelling pipeline and the token ids can
                # supply. No GPU, so this one runs anywhere.
                "python scripts/score_baselines.py --build-root $BUILD --out-dir outputs/baselines",
                "for b in repetition repetition_trailing rep_l lrs entropy; do"
                " python scripts/evaluate_scores.py --run-dir outputs/baselines/$b; done",
                # Activation self-similarity needs the cached hidden states, so
                # it reads roughly 30 GB per split and goes through Slurm.
                "sbatch cluster/activation_similarity.sbatch --split val",
                "for d in outputs/baselines/actsim_*; do"
                " python scripts/evaluate_scores.py --run-dir $d --splits val; done",
                # What persistence would buy, on validation only.
                "python scripts/evaluate_scores.py --run-dir outputs/baselines/repetition_trailing"
                " --splits val --compare-persistence 1 4 16 64",
            ],
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
            "axes": ["features", "selection", "window", "horizon", "layer"],
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

    return load_baseline_view, load_view, nothing_scored, scoring_progress


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
        ("C", "fired before the loop", "view_c_lead_time", "fired_before_frontier", "share of caught answers flagged early"),
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
        """Whether a configuration name belongs to a model-free scorer.

        Matched by prefix rather than by an explicit list, so a scorer added to
        `outputs/baselines/` appears on every table without being registered in
        two places. The activation-similarity variants carry their layer and
        window in the name, which is why they cannot be enumerated here.
        """
        return name.startswith(("repetition", "entropy", "lrs", "rep_l", "actsim"))

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

    return budget_control, layer_control, test_view_controls, view_controls


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
def _(BUILD_ROOT, REPO, pd):
    def evaluation_population():
        """How many answers the pinned validation population holds, by class.

        The replay records rates, and turning a rate back into a count is what
        precision needs. Read from the pinned file itself rather than written
        down here, so it cannot drift from what the runs actually measured.
        """
        pinned = sorted((REPO / "configs" / "dataset").glob("validation_rollouts_*.csv"))
        labels_path = BUILD_ROOT / "onset_labels" / "onset_labels.parquet"
        if not pinned or not labels_path.is_file():
            return 0, 0
        chosen = pd.read_csv(pinned[0])
        labels = pd.read_parquet(
            labels_path, columns=["domain", "prompt_id", "rollout_idx", "is_positive"]
        )
        merged = chosen.merge(
            labels, on=["domain", "prompt_id", "rollout_idx"], how="left"
        )
        flags = merged["is_positive"].fillna(False).astype(bool)
        return int(flags.sum()), int((~flags).sum())

    EVAL_POSITIVES, EVAL_NEGATIVES = evaluation_population()
    return EVAL_NEGATIVES, EVAL_POSITIVES


@app.cell
def _(
    EVAL_NEGATIVES,
    EVAL_POSITIVES,
    apply_stopping_rule,
    missing,
    mo,
    pd,
    stage_replay,
):
    # The replay measured one operating point, so this table is fixed at it.
    # Everything a different budget would need from the degenerate answers was
    # reduced to these numbers and not kept, so 5% and 10% are not a filter away.
    REPLAY_BUDGET = 0.01
    RULE_STEP = "the step the rule picks"
    BEST_DEPTH = "the best depth"

    def replay_controls(exp_id):
        """One stage's own step and depth pickers for the table below it."""
        panel = stage_replay(exp_id)
        steps = (
            {str(int(s)): int(s) for s in sorted(panel["step"].unique())}
            if not panel.empty
            else {}
        )
        depths = (
            {str(int(d)): int(d) for d in sorted(panel["layer"].unique())}
            if not panel.empty
            else {}
        )
        return mo.ui.dictionary(
            {
                "step": mo.ui.dropdown(
                    options={RULE_STEP: RULE_STEP, **steps},
                    value=RULE_STEP,
                    label="checkpoint",
                ),
                "depth": mo.ui.dropdown(
                    options={BEST_DEPTH: BEST_DEPTH, **depths},
                    value=BEST_DEPTH,
                    label="depth",
                ),
            }
        )

    def _one_run(block, step_choice, depth_choice):
        """The (depth, step) one run is read at, given the two choices."""
        if step_choice == RULE_STEP:
            outcomes = apply_stopping_rule(block)
            eligible = outcomes[outcomes["selected_value"].notna()]
            if eligible.empty:
                return None
            if depth_choice == BEST_DEPTH:
                pick = eligible.loc[eligible["selected_value"].idxmax()]
                return int(pick["layer"]), int(pick["selected_step"])
            at_depth = eligible[eligible["layer"] == depth_choice]
            if at_depth.empty:
                return None
            return int(depth_choice), int(at_depth.iloc[0]["selected_step"])
        at_step = block[block["step"] == step_choice]
        if at_step.empty:
            return None
        if depth_choice == BEST_DEPTH:
            best = at_step.loc[at_step["warning_recall_256"].idxmax()]
            return int(best["layer"]), int(step_choice)
        return int(depth_choice), int(step_choice)

    def replay_records(exp_id, step_choice, depth_choice):
        """One row per seed run: where it was read, and what it scored there."""
        panel = stage_replay(exp_id)
        if panel.empty:
            return pd.DataFrame()
        rows = []
        for (configuration, run), block in panel.groupby(["configuration", "run"]):
            where = _one_run(block, step_choice, depth_choice)
            if where is None:
                continue
            layer, step = where
            found = block[(block["layer"] == layer) & (block["step"] == step)]
            if found.empty:
                continue
            record = found.iloc[0]
            caught = float(record["recall_at_budget"]) * EVAL_POSITIVES
            false_alarms = float(record["budget_realized_fpr"]) * EVAL_NEGATIVES
            flagged = caught + false_alarms
            rows.append(
                {
                    "configuration": configuration,
                    "run": run,
                    "depth": layer,
                    "step": step,
                    "recall": float(record["recall_at_budget"]),
                    # Not stored by the replay, but implied by it: the healthy
                    # answers that fired are the realised false-alarm rate times
                    # their number, and the degenerate ones that fired are the
                    # recall times theirs.
                    "precision": (caught / flagged) if flagged else float("nan"),
                    "in-pattern coverage": float(record["in_pattern_recall"]),
                    "warning coverage 128": float(record["warning_recall_128"]),
                    "warning coverage 256": float(record["warning_recall_256"]),
                    "never fired": float(record["never_fired_positives"]),
                    "median lead": float(record["median_offset"]),
                }
            )
        return pd.DataFrame(rows)

    def _spread(values, digits=3):
        """The median across seeds, with the seed-to-seed range beside it."""
        values = values.dropna()
        if values.empty:
            return ""
        middle = values.median()
        text = f"{middle:.{digits}f}"
        if len(values) > 1 and values.min() != values.max():
            return f"{text} [{values.min():.{digits}f}–{values.max():.{digits}f}]"
        return text

    def _whole(values):
        return _spread(values, digits=0)

    REPLAY_COLUMNS = [
        "recall",
        "precision",
        "in-pattern coverage",
        "warning coverage 128",
        "warning coverage 256",
        "never fired",
        "median lead",
    ]

    def replay_scorecard(exp_id, step_choice, depth_choice):
        """Every configuration of a stage, from its replayed checkpoints."""
        long = replay_records(exp_id, step_choice, depth_choice)
        if long.empty:
            return missing(
                f"No run of {exp_id} has had its checkpoints replayed, so there "
                "is nothing to read here. Adapted runs cannot be replayed "
                "cheaply, which is why S3 is empty."
            )
        rounding = {name: _whole if name in {"never fired", "median lead"} else _spread
                    for name in REPLAY_COLUMNS}
        table = (
            long.groupby("configuration")
            .agg(
                seeds=("run", "nunique"),
                depth=("depth", _whole),
                step=("step", _whole),
                **{name: (name, rounding[name]) for name in REPLAY_COLUMNS},
            )
            .reset_index()
        )
        order = (
            long.groupby("configuration")["warning coverage 256"].median().sort_values(
                ascending=False
            )
        )
        table = table.set_index("configuration").loc[order.index].reset_index()
        where = (
            "each run at the checkpoint its stopping rule selects"
            if step_choice == RULE_STEP
            else f"every run at step {step_choice}"
        )
        depth_note = (
            "each read at whichever depth scores best there"
            if depth_choice == BEST_DEPTH
            else f"all read at layer {depth_choice}"
        )
        return mo.vstack(
            [
                mo.md(
                    f"**Every configuration of {exp_id}, including the ones that "
                    "were not carried forward.** One row per configuration, "
                    f"{where}, {depth_note}. Each cell is the median across the "
                    "configuration's seeds with the seed-to-seed range in "
                    "brackets, so a column whose brackets all overlap is not "
                    "ranking anything. Sorted by warning coverage over 256 "
                    "tokens, the quantity the project selects on."
                ),
                mo.ui.table(table, selection=None),
                mo.md(
                    f"_Read from `checkpoint_replay.parquet`, so this covers every "
                    f"configuration rather than only the few that have been "
                    f"through a full scoring pass. The price is that the replay "
                    f"measured a single operating point: everything here is at a "
                    f"**{REPLAY_BUDGET:.0%} false-alarm budget** and cannot be "
                    "moved to another one, because the per-token scores it was "
                    "computed from were not kept. **Precision** is derived from "
                    "the recorded recall and realised false-alarm rate against "
                    f"the pinned population of {EVAL_POSITIVES} degenerate and "
                    f"{EVAL_NEGATIVES} healthy answers. The healthy token rate "
                    "and the persistence view are absent for the same reason, "
                    "and appear only in the full-protocol table below._"
                ),
            ]
        )

    return replay_controls, replay_scorecard


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

    Before asking what a probe can do, it is worth knowing what can be done
    without training anything. Every scorer here goes through the identical
    evaluator, sees the same answers, gets its thresholds frozen the same way,
    and is reported in the same four views.

    Which window a scorer reads is the axis that separates them, because a
    monitor running alongside generation cannot see tokens that do not exist yet.

    - **repetition**, one minus the type-token ratio over bigrams on a 256-token
      window placed *ahead* of the token it labels. This is what the labelling
      pipeline computes, and it is reported for reference only. No live system
      could run it.
    - **repetition_trailing**, the same statistic over the 256 tokens *ending* at
      the token it labels. This is the causal form, and it is the one every
      comparison uses.
    - **rep_l**, the per-token repetition rate of the generation literature,
      backward looking by construction, at l = 128.
    - **lrs**, the longest repeated substring, as a step at the position the
      repeat begins.
    - **entropy**, the model's own predictive entropy averaged over the trailing
      128 tokens and inverted, on the grounds that a loop is confident rather
      than uncertain.
    - **actsim**, the cosine similarity between the current hidden state and the
      most similar recent one. Model-internal but untrained, so it is the sharpest
      floor for a probe: it asks whether a learned direction beats simply noticing
      that the residual stream has begun revisiting where it has been.

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
    **Before the table: which scorers can be operated at all.**

    An answer is predicted positive when its first alarm is finite, which at
    persistence 1 means its highest-scoring token reaches the threshold. So the
    share of *healthy* answers tied at the very top of a scorer's range is a hard
    floor on the false-alarm rate that scorer can be asked for. When that share
    exceeds the budget, no threshold isolates a small enough slice: the solver
    pushes the threshold past the top of the range and nothing fires anywhere,
    positives included.

    Those rows read zero across the board, and the accuracy column still reads
    0.9703, which is just the share of the split that is negative. A row of
    zeros beside a respectable accuracy is the signature of a scorer that never
    fired, not of one that was almost right.

    The evaluator reports this directly rather than leaving it to be inferred:
    each entry in `decision_thresholds.json` carries `tied_at_ceiling` and a
    `saturated` flag.

    | scorer | healthy tied at ceiling | 1% | 5% | 10% |
    |---|---|---|---|---|
    | `repetition` (forward) | 0.03% | yes | yes | yes |
    | `repetition_trailing` | 0.03% | yes | yes | yes |
    | `rep_l`, l = 128 | 1.53% | **no** | yes | yes |
    | `lrs` | 81.06% | **no** | **no** | **no** |
    | `entropy`, trailing 128 | 0.03% | yes | yes | yes |

    `lrs` is a step function to exactly 1.0, so four fifths of healthy answers
    reach its ceiling and it cannot be operated at any budget in use here. It
    stays on the table as the reason the protocol reports this quantity at all.
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
    mo.md(r"""
    ### Why the entropy baseline needs a window

    Inverted entropy is $1/(1+H)$, which is monotone decreasing in $H$ and lands
    in $(0, 1]$ without needing a population to normalise against. Read per token
    it is unusable, for a reason that has nothing to do with degeneration: a
    single token drawn at near-zero entropy is completely ordinary in healthy
    text. A closing bracket, the second half of a tokenised word, the digit after
    a decimal point. Every one of those is near-certain, and one is enough to put
    an answer at the top of the range.

    On validation, **98.6% of healthy answers** contain a token whose entropy is
    exactly zero to machine precision, so 98.6% tie at the ceiling and no budget
    can be spent at any level.

    Two things fix it, and both were needed.

    **Average before inverting.** The score becomes $1/(1 + \bar{H}_w)$, where
    $\bar{H}_w$ is the mean entropy over the trailing $w$ tokens. This asks
    whether the model has been confident for a while, which is the property a
    loop actually has, rather than whether it was ever confident once. Smoothing
    the entropy and inverting afterwards is preferred over the reverse only
    because the average is then taken on the quantity the model reports, in nats.
    The two orders agree to four decimal places.

    The width is not a free parameter. It is the narrowest one that brings the
    tied share under a 1% budget, which also happens to be the lookback `rep_l`
    already uses, so the two per-token signals share a scale.

    | trailing width | healthy tied at ceiling | 1% budget spendable |
    |---|---|---|
    | none (per token) | 98.56% | no |
    | 32 | 4.78% | no |
    | 64 | 1.70% | no |
    | **128** | **0.23%** | **yes** |
    | 256 | 0.06% | yes |

    **Store the scores at single precision.** Half precision has a spacing of
    about $5 \times 10^{-4}$ just below one, so every score within that distance
    of the top of the range collapses onto it. That is enough to move the
    windowed entropy score from 0.23% of healthy answers tied at its ceiling to
    1.25%, which is the difference between a 1% budget that can be spent and one
    that cannot. Ties at a ceiling are exactly what the protocol reports as a
    failure, so storing at half precision manufactures the failure. Scores are
    written as `float32`.

    Nothing else on the table was close enough to the ceiling for the storage
    width to matter. The probe scores never approach it, and the ties in `rep_l`
    and `lrs` are real: `rep_l` is a mean of indicators that can be exactly one,
    and `lrs` is a literal step to one.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What an answer-level number is measuring on this corpus

    Every positive is by definition an answer that reached the 4096-token cap,
    and healthy answers have a median length of 475. Any rollout score formed as
    a maximum over tokens therefore rises with length, and the classes separate
    before any model is consulted.

    The size of this is easy to underestimate. **Answer length alone reaches an
    answer-level AUC of 0.9992 on validation.** The windowed entropy score
    reaches 0.9917 at $w = 128$ and 0.9960 at $w = 256$, so on the whole
    population it scores slightly *worse* than a ruler.

    Two controls separate length from what a scorer knows.

    Restricting the healthy population to long answers weakens the effect but
    does not remove it, and length still wins on every one of those populations:

    | healthy answers kept | windowed entropy | length alone |
    |---|---|---|
    | all | 0.9843 | 0.9982 |
    | at least 1000 tokens | 0.9588 | 0.9918 |
    | at least 1500 tokens | 0.9167 | 0.9639 |
    | at least 2000 tokens | 0.7528 | 0.8846 |

    Judging every answer on an equal-length prefix removes it entirely. Scored on
    its first 128, 256 or 474 tokens, the windowed entropy score reaches AUC 0.45,
    0.48 and 0.53. That is chance.

    **What this licenses, and what it does not.** It is a consequence of how the
    label is built rather than something learned about degeneration, it would move
    with the cap, and no manual inspection has been done. It does not support the
    claim that length is uninformative about degeneration. The defensible
    statement is narrow: under this label definition an answer-level ranking is
    partly a ranking by length, so it must not be used to choose a threshold, a
    depth or a recipe.

    Answer-level figures are still reported, because the paper claims they are
    saturated and that claim has to be made with the quantity in hand. Which of
    them is saturated and which is not turns out to depend on the metric, and is
    measured in Part 6: accuracy and AUC are, average precision is not, and
    accuracy on a balanced set is inverted. Nothing is ever selected on any of
    them.

    This does not touch the token-level work. Warning coverage is measured inside
    degenerate answers against healthy tokens at matched absolute position, so it
    already controls for the confound above. Keeping that separation visible is
    the whole reason coverage is split at the frontier rather than pooled.

    **The same gate appears in the literature.** Two of the four recent loop
    detectors require the generation to reach its token limit before a loop
    counts: Yu et al. only score a response that hits `max_new_tokens`, and
    LoopGuard requires length at least 2480 against a 2500-token maximum. Duan et
    al. and Xie et al. do not. So this is a caution about the measurement
    conventions of the area, not about this corpus alone.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Activation self-similarity: the floor a probe really has to clear

    A repetition counter is a floor, but a weak one, because it reads the text
    rather than the model. The sharper question is whether a *trained* direction
    beats simply noticing that the model's own hidden states have started
    revisiting where they have already been.

    That needs no training. At each token, take the cosine similarity between the
    current hidden state and each of the previous ones inside a trailing window,
    and keep the largest. It is scored through the same interface as everything
    else, so it lands in the same four views.

    Read at one layer, so the cost is one strided read per rollout rather than a
    forward pass. The cache stores `[33, tokens, 4096]` per answer with the layer
    axis outermost, so a single layer is one contiguous block and asking for six
    of them costs little more than asking for one.

    **This is not Yu et al.'s detector, and should not be named for it.** Theirs
    collects per-token maxima of activation similarity across the decoder's MLP
    layers, sorts them, concatenates the sorted and unsorted vectors, and passes
    the result to a three-layer network that scores a whole response after a
    400-token warm-up. What is shared is the primitive, not the method.

    The minimum lag matters and is swept rather than assumed. Neighbouring
    residual streams are similar for reasons unrelated to a loop, so a scorer
    comparing a token only to its immediate predecessor reports that similarity
    everywhere. Lags of 1, 16 and 64 are computed in the same pass.

    One detail worth not rediscovering: the first `min_lag` tokens have nothing
    behind them to be similar to, and they are scored at the *bottom* of the
    range rather than at zero similarity. Zero cosine maps to the middle of the
    unit interval, which would make every answer alarm on its own opening tokens.

    On a three-answer smoke test the signal is real: healthy answers sit at a
    median of 0.91, and inside a loop the median is 0.992, with the pre-onset
    stretch of the same answer at 0.905.
    """)
    return


@app.cell
def _(BASELINE_ROOT, mo, pd):
    def persistence_sweep(scorer="repetition_trailing"):
        """What requiring m consecutive tokens above threshold buys."""
        path = BASELINE_ROOT / scorer / "evaluation" / "persistence_comparison.csv"
        if not path.is_file():
            return mo.md(
                f"_`{scorer}` has no persistence comparison yet. Run "
                "`evaluate_scores.py --compare-persistence 1 4 16 64`._"
            )
        frame = pd.read_csv(path)
        keep = [
            column
            for column in (
                "persistence",
                "target_negative_fpr",
                "tau",
                "median_offset",
                "never_fired_positives",
            )
            if column in frame.columns
        ]
        return mo.ui.table(frame[keep], selection=None)

    mo.vstack(
        [
            mo.md(
                """
                ### What persistence would buy

                Every number reported anywhere in this project uses persistence
                $m = 1$: one token at or above the threshold is an alarm. That is
                the most twitchy rule available, and it is a stated choice rather
                than an assumption, so it is worth knowing what the alternative
                is worth.

                Requiring $m$ consecutive tokens above the threshold lets the
                threshold itself relax at the same false-alarm budget, because a
                sustained run is rarer than a spike. The alarm then arrives
                *earlier*, not later.
                """
            ),
            persistence_sweep(),
            mo.md(
                """
                At a 10% budget the trailing repetition score's median alarm moves
                from 31 tokens after the frontier at $m = 1$ to 6 tokens before it
                at $m = 64$. `rep_l` gains something else: at $m = 64$ it can
                spend a 1% budget for the first time, because requiring 64
                consecutive tokens above threshold breaks the ties at its ceiling
                that block it at $m = 1$.

                So $m = 1$ is conservative rather than flattering. Persistence
                belongs in the appendix as a sweep, and the main results stay at
                $m = 1$ so that one fewer thing varies between them.
                """
            ),
        ]
    )
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
def _(replay_controls):
    s1_replay_controls = replay_controls("S1")
    s1_replay_controls
    return (s1_replay_controls,)


@app.cell
def _(replay_scorecard, s1_replay_controls):
    replay_scorecard(
        "S1",
        s1_replay_controls.value["step"],
        s1_replay_controls.value["depth"],
    )
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
def _(
    FIGURES,
    INK_SOFT,
    RAMP,
    apply_stopping_rule,
    missing,
    mo,
    mpl,
    pd,
    plt,
    save,
    stage_replay,
    tidy,
):
    def depth_profile_figure(exp_id):
        """Where in the network the approach becomes visible, per selection rule.

        Read from the replayed checkpoints rather than from full scoring passes.
        Full scoring exists at one depth per run, which cannot answer a question
        asked about depth; the replay carries every depth of every run.
        """
        panel = stage_replay(exp_id)
        if panel.empty:
            return missing(
                f"No run of {exp_id} has had its checkpoints replayed, so there "
                "is no depth profile to draw."
            )
        rows = []
        for _, block in panel.groupby("run"):
            outcomes = apply_stopping_rule(block)
            rule = block["rule"].iloc[0]
            width = block["window"].iloc[0]
            for picked in outcomes.itertuples():
                if pd.isna(picked.selected_step):
                    continue
                at_step = block[
                    (block["layer"] == picked.layer)
                    & (block["step"] == picked.selected_step)
                ]
                if at_step.empty:
                    continue
                rows.append(
                    {
                        "rule": rule,
                        "width": width,
                        "layer": int(picked.layer),
                        "warning": float(at_step["warning_recall_256"].iloc[0]),
                    }
                )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return missing("No depth of this stage ever became selectable.")

        rules = sorted(frame["rule"].dropna().unique())
        widths = sorted(int(w) for w in frame["width"].dropna().unique())
        shades = mpl.colors.LinearSegmentedColormap.from_list("width", RAMP)
        scale = mpl.colors.Normalize(
            vmin=min(widths), vmax=max(widths)
        ) if len(widths) > 1 else None

        fig, axes = plt.subplots(
            1, len(rules), figsize=(2.9 * len(rules), 3.4), squeeze=False, sharey=True
        )
        for axis, rule in zip(axes[0], rules):
            block = frame[frame["rule"] == rule]
            for width, group in block.groupby("width", dropna=False):
                line = group.groupby("layer")["warning"].median().sort_index()
                if line.empty:
                    continue
                if pd.isna(width):
                    # Two of the rules tile or sample instead of placing a
                    # window, so they have one line and no width to encode.
                    colour, label = RAMP[2], "no window"
                else:
                    colour = shades(scale(int(width))) if scale else RAMP[2]
                    label = f"trained on {int(width)}-token windows"
                axis.plot(
                    line.index, line.to_numpy(), "o-", color=colour,
                    markersize=3, linewidth=1.4, label=label,
                )
            axis.set_title(rule, color=INK_SOFT, fontsize=8)
            axis.set_xlabel("layer")
            tidy(axis, xgrid=False)
        # The vertical axis is a width too, which is exactly the confusion this
        # figure has to avoid: the panels vary the window a probe was *trained*
        # on, the axis counts the tokens it is *scored* over.
        axes[0][0].set_ylabel("warning coverage\n(256 tokens before the loop)")

        seen, handles, names = set(), [], []
        for axis in axes[0]:
            for handle, name in zip(*axis.get_legend_handles_labels()):
                if name not in seen:
                    seen.add(name)
                    handles.append(handle)
                    names.append(name)
        fig.legend(
            handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(names), 5),
        )
        fig.suptitle(
            f"{exp_id}: where the approach is visible, by depth "
            "(validation, 1% false-alarm budget)",
            x=0.005, ha="left", fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.92))
        return mo.vstack(
            [
                save(fig, f"{exp_id}_depth_profile", FIGURES),
                mo.md(
                    "_**Two different token counts meet in this figure, so they "
                    "are named apart.** The panels split the rules by the window "
                    "size each probe was *trained* on; the vertical axis is "
                    "warning coverage, which is always measured over the 256 "
                    "tokens immediately before the loop, whatever a run was "
                    "trained on. One line per training width, one point per "
                    "depth, median across the three seeds._\n\n"
                    "_A line stops where its depths stop clearing the "
                    "in-loop-coverage floor, so a short line is a result rather "
                    "than missing data. It is the **anchored** rules that lose "
                    "depths to a narrow window: at 64 tokens `frontier_window` "
                    "and its hard-negative variant leave only 11 and 12 of the "
                    "31 depths ever selectable, all between layers 6 and 18, "
                    "rising to 20 depths at 128 and 27 at 256 and above. "
                    "`random_window` at the same 64 tokens keeps 29, because its "
                    "windows are scattered through the answer rather than spent "
                    "around the frontier. Note that the width means different "
                    "things across panels: for the anchored rules it is a window "
                    "placed on the frontier, for `rollout_balanced` it is how "
                    "many tokens each answer contributes._\n\n"
                    "_Drawn from the replayed checkpoints, each depth read at "
                    "the checkpoint its own stopping rule selects, which is why "
                    "every configuration has a full profile rather than the "
                    "single scored depth a full protocol pass can afford._"
                ),
            ]
        )

    depth_profile_figure("S1")
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
def _(replay_controls):
    s2a_replay_controls = replay_controls("S2a")
    s2a_replay_controls
    return (s2a_replay_controls,)


@app.cell
def _(replay_scorecard, s2a_replay_controls):
    replay_scorecard(
        "S2a",
        s2a_replay_controls.value["step"],
        s2a_replay_controls.value["depth"],
    )
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
def _(replay_controls):
    s2b_replay_controls = replay_controls("S2b")
    s2b_replay_controls
    return (s2b_replay_controls,)


@app.cell
def _(replay_scorecard, s2b_replay_controls):
    replay_scorecard(
        "S2b",
        s2b_replay_controls.value["step"],
        s2b_replay_controls.value["depth"],
    )
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
def _(replay_controls):
    s2c_replay_controls = replay_controls("S2c")
    s2c_replay_controls
    return (s2c_replay_controls,)


@app.cell
def _(replay_scorecard, s2c_replay_controls):
    replay_scorecard(
        "S2c",
        s2c_replay_controls.value["step"],
        s2c_replay_controls.value["depth"],
    )
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
def _(replay_controls):
    s2d_replay_controls = replay_controls("S2d")
    s2d_replay_controls
    return (s2d_replay_controls,)


@app.cell
def _(replay_scorecard, s2d_replay_controls):
    replay_scorecard(
        "S2d",
        s2d_replay_controls.value["step"],
        s2d_replay_controls.value["depth"],
    )
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
    #
    # Both halves of that pick are hand-made, and neither is recorded in the run
    # directories, which is what the table two cells below exists to show. The
    # depth is written out here by hand; the checkpoint that filled
    # `layers/layer_NN/` is not written down at all. Scoring records its own
    # provenance now, so directories written from here on can say what produced
    # them, and `scripts/audit_score_provenance.py` reports which of the
    # existing ones cannot.
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
        ("C", "fired before the loop", "view_c_lead_time", "fired_before_frontier"),
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
                    "automatic per-run checkpoint the rest of the notebook uses. "
                    "Both the depth and the checkpoint here are hand-picked, and "
                    "the table below says how far each sits from what the rule "
                    f"would have chosen. At a **{budget:.0%} false-alarm budget** "
                    f"on `val`.{progress}"
                ),
                mo.ui.table(table, selection=None),
            ]
        )

    candidate_scorecard(candidates_card_budget.value)
    return


@app.cell
def _(
    DEFAULT_RULE,
    LEADING_CANDIDATES,
    Path,
    all_runs,
    apply_stopping_rule,
    missing,
    mo,
    pd,
    replay_frame,
):
    def hand_pick_against_rule():
        """Where each candidate was read, beside where the rule would read it.

        Two choices decide what a candidate's row says: the depth, and the
        checkpoint within that depth. The rule owns the second and, applied
        across depths, implies the first. The list above sets both by hand, so
        the two can disagree, and a disagreement is worth seeing rather than
        assuming away.
        """
        if replay_frame.empty or all_runs.empty:
            return missing("No run has had its checkpoints replayed yet.")
        rows = []
        for candidate in LEADING_CANDIDATES:
            matches = all_runs[
                (all_runs["group"] == candidate["group"])
                & (all_runs["status"] == "finished")
            ]
            for run in matches.itertuples():
                panel = replay_frame[
                    replay_frame["run"] == Path(run.run_dir).parent.name
                ]
                if panel.empty:
                    continue
                outcomes = apply_stopping_rule(panel)
                eligible = outcomes[outcomes["selected_value"].notna()]
                if eligible.empty:
                    continue
                best = eligible.loc[eligible["selected_value"].idxmax()]
                chosen = eligible[eligible["layer"] == candidate["depth"]]
                at_hand_pick = (
                    float(chosen.iloc[0]["selected_value"]) if not chosen.empty else None
                )
                rows.append(
                    {
                        "candidate": candidate["label"],
                        "seed": run.seed,
                        "depth, hand-picked": candidate["depth"],
                        "depth, by the rule": int(best["layer"]),
                        "step, by the rule": int(best["selected_step"]),
                        "warning coverage at the rule's depth": round(
                            float(best["selected_value"]), 4
                        ),
                        "warning coverage at the hand-picked depth": (
                            round(at_hand_pick, 4) if at_hand_pick is not None else None
                        ),
                    }
                )
        if not rows:
            return missing("No candidate has a replayed run to compare against.")
        frame = pd.DataFrame(rows)
        frame["depths agree"] = (
            frame["depth, hand-picked"] == frame["depth, by the rule"]
        )
        agreed = int(frame["depths agree"].sum())
        return mo.vstack(
            [
                mo.md(
                    f"One row per candidate per seed. The rule of "
                    f"`head_selection.py` (floor {DEFAULT_RULE.floor}, band "
                    f"{DEFAULT_RULE.band}, tolerance {DEFAULT_RULE.tolerance}, "
                    f"patience {DEFAULT_RULE.patience}) keeps the best checkpoint "
                    "of each depth; the depth it implies is then the one whose "
                    "kept checkpoint has the highest warning coverage. "
                    f"**The two agree on {agreed} of {len(frame)} runs.**"
                ),
                mo.ui.table(frame, selection=None),
                mo.md(
                    "Where they disagree the cost is small: the largest gap in "
                    "warning coverage is under a hundredth, and all but one of the "
                    "disagreements stay inside layers 12 to 16, the band the depth "
                    "profile already marks out. The exception is worth naming, "
                    "since it is the case the rest of the table would hide: on one "
                    "seed of `frontier_window` at W=256 the rule prefers layer 5 "
                    "over layer 15 outright. So the depth is not knife-edge in "
                    "general, and a hand-pick near the top of the band usually "
                    "loses little, but not on every run. What none of this "
                    "licenses is calling the pick reproducible. Neither the depth above nor the checkpoint "
                    "behind the scores is recorded anywhere, and the checkpoint "
                    "cannot be recovered from the stored numbers, because "
                    "neighbouring steps of one depth differ by less than the "
                    "precision the replay carries. Before the held-out test is "
                    "read, the rule should choose both, once, and the choice "
                    "should be written down with the scores it produced."
                ),
            ]
        )

    hand_pick_against_rule()
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
    ### Aside: which domain is this, and how much of the warning is that?

    The table above reports one warning coverage per candidate, pooled over
    every degenerate answer in the validation split. Pooling treats the five
    in-domain sources as five samples of one population. They are not, and the
    gap between them is wide enough that the pooled figure describes none of
    them.

    Three readings follow. The first splits the same twelve candidates by the
    domain their prompt came from. The second asks whether that split is really
    about the domain, or only about how far into an answer each domain's loops
    happen to begin, since a probe's score climbs with position on its own. The
    third asks the same question of the false alarms, where the confound to
    rule out is answer length rather than position.
    """)
    return


@app.cell
def _(LEADING_CANDIDATES, Path, all_runs, pd):
    # Below this many degenerate answers a domain's rates are reported and
    # marked anecdotal, never drawn as a point on a curve, since one answer
    # moving decides the whole cell.
    DOMAIN_MINIMUM = 10

    def domain_rollouts(label, budget):
        """Every scored answer of one candidate, at every seed it was run at."""
        candidate = next(c for c in LEADING_CANDIDATES if c["label"] == label)
        if all_runs.empty:
            return pd.DataFrame()
        matches = all_runs[
            (all_runs["group"] == candidate["group"])
            & (all_runs["status"] == "finished")
        ]
        frames = []
        for run in matches.itertuples():
            path = (
                Path(run.run_dir)
                / "layers"
                / f"layer_{candidate['depth']:02d}"
                / "evaluation"
                / "val"
                / "rollout_table.csv"
            )
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame = frame[frame["target_negative_fpr"] == budget].copy()
            frame["seed"] = run.seed
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def domain_rates(table):
        """One row per domain: how much run-up is flagged, how early, how often wrongly.

        Coverage is pooled over tokens rather than averaged over answers, the
        same way the protocol computes it, so a long answer carries the weight
        its token count earns. Answer counts are divided by the number of seeds,
        since each seed contributes the same answers over again.
        """
        if table.empty:
            return pd.DataFrame()
        positives = table[table["is_positive"]]
        negatives = table[~table["is_positive"]]
        seeds = max(int(table["seed"].nunique()), 1)
        all_warning_hits = positives["warning_hits_256"].sum()
        rows = []
        for domain in sorted(table["domain"].unique()):
            degenerate = positives[positives["domain"] == domain]
            healthy = negatives[negatives["domain"] == domain]
            fired = degenerate[degenerate["fired"].astype(bool)]
            warning_tokens = degenerate["warning_tokens_256"].sum()
            in_pattern_tokens = degenerate["in_pattern_tokens"].sum()
            rows.append(
                {
                    "domain": domain,
                    "degenerate answers": len(degenerate) // seeds,
                    "healthy answers": len(healthy) // seeds,
                    "median onset": degenerate["onset_position"].median(),
                    "warning coverage 256": (
                        degenerate["warning_hits_256"].sum() / warning_tokens
                        if warning_tokens
                        else None
                    ),
                    "share of all warning hits": (
                        degenerate["warning_hits_256"].sum() / all_warning_hits
                        if all_warning_hits
                        else None
                    ),
                    "in-pattern coverage": (
                        degenerate["in_pattern_hits"].sum() / in_pattern_tokens
                        if in_pattern_tokens
                        else None
                    ),
                    "median lead": (
                        fired["offset"].median() if len(fired) else None
                    ),
                    "fired before the loop": (
                        float((fired["offset"] < 0).mean()) if len(fired) else None
                    ),
                    "false alarms": int(healthy["fired"].astype(bool).sum()) // seeds,
                }
            )
        return pd.DataFrame(rows)

    return DOMAIN_MINIMUM, domain_rates, domain_rollouts


@app.cell
def _(budget_control, candidate_control, mo):
    domain_pick = candidate_control()
    domain_budget = budget_control()
    mo.hstack([domain_pick, domain_budget], justify="start", gap=2)
    return domain_budget, domain_pick


@app.cell
def _(domain_budget, domain_pick, domain_rates, domain_rollouts, missing, mo):
    def domain_table(label, budget):
        rates = domain_rates(domain_rollouts(label, budget))
        if rates.empty:
            return missing("This candidate has not been scored on `val` yet.")
        return mo.vstack(
            [
                mo.md(
                    f"**{label}**, pooled over its three seeds, at a "
                    f"**{budget:.0%} false-alarm budget** on `val`. One row per "
                    "in-domain source. **Share of all warning hits** is what "
                    "fraction of every flagged run-up token in the whole split "
                    "came from this domain, which is the number that says how "
                    "much of the headline figure one domain is carrying. "
                    "`aime_2025` has a single degenerate answer and is "
                    "anecdotal in every column."
                ),
                mo.ui.table(rates, selection=None),
            ]
        )

    domain_table(domain_pick.value, domain_budget.value)
    return


@app.cell
def _(
    DOMAIN_MINIMUM,
    FIGURES,
    INK_MUTED,
    INK_SOFT,
    LEADING_CANDIDATES,
    SERIES,
    STAGE_ORDER,
    domain_budget,
    domain_rates,
    domain_rollouts,
    missing,
    mo,
    pd,
    plt,
    save,
    tidy,
):
    def domain_profile_figure(budget):
        """Every candidate's two coverages, against the domain, on one axis.

        One candidate could put its warning coverage anywhere by chance. Twelve
        of them, from four stages, agreeing on the order of the domains is a
        statement about the corpus instead.
        """
        frames = []
        for candidate in LEADING_CANDIDATES:
            rates = domain_rates(domain_rollouts(candidate["label"], budget))
            if rates.empty:
                continue
            rates["label"] = candidate["label"]
            rates["stage"] = candidate["stage"]
            frames.append(rates)
        if not frames:
            return missing("No candidate has been scored on `val` yet.")
        long = pd.concat(frames, ignore_index=True)
        long = long[long["degenerate answers"] >= DOMAIN_MINIMUM]
        if long.empty:
            return missing("No domain has enough degenerate answers to plot.")

        # Domains are ordered by where their loops start, so the figure can be
        # read against the position confound the next figure tests.
        order = (
            long.groupby("domain")["median onset"].median().sort_values().index.tolist()
        )
        position = {domain: index for index, domain in enumerate(order)}
        colors = dict(zip(STAGE_ORDER, SERIES[: len(STAGE_ORDER)]))

        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
        panels = [
            ("warning coverage 256", "coverage of the 256 before the loop"),
            ("in-pattern coverage", "coverage inside the loop"),
        ]
        for axis, (column, ylabel) in zip(axes, panels):
            for label, block in long.groupby("label"):
                block = block.dropna(subset=[column])
                if block.empty:
                    continue
                block = block.assign(x=block["domain"].map(position)).sort_values("x")
                axis.plot(
                    block["x"],
                    block[column],
                    "o-",
                    color=colors[block["stage"].iloc[0]],
                    linewidth=1.1,
                    markersize=4,
                    alpha=0.85,
                )
            axis.set_xticks(range(len(order)))
            axis.set_xticklabels(order, rotation=30, ha="right", fontsize=7.5)
            axis.set_ylabel(ylabel)
            axis.set_ylim(bottom=0)
            tidy(axis, xgrid=False)
        axes[0].set_xlabel("domain, ordered by where its loops begin", color=INK_SOFT)
        axes[1].set_xlabel("domain, ordered by where its loops begin", color=INK_SOFT)

        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=colors[stage], label=stage)
            for stage in STAGE_ORDER
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(STAGE_ORDER),
            fontsize=8,
            title="stage",
            title_fontsize=8,
        )
        fig.suptitle(
            f"The same twelve candidates, split by domain, at a {budget:.0%} "
            "false-alarm budget",
            x=0.005,
            ha="left",
            fontsize=10,
            color=INK_MUTED,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        return mo.vstack(
            [
                save(fig, "candidate_domain_profile", FIGURES),
                mo.md(
                    "_One line per leading candidate, coloured by the stage it "
                    "came from. **The two panels rank the domains differently, "
                    "and that is the whole point.** Left, coverage of the run-up: "
                    "`llama_nemotron` is the highest domain in every one of the "
                    "twelve candidates and `deepmath_103k` sits near zero, a "
                    "spread of twenty to sixty fold within a single candidate. "
                    "Code carries between a half and seven eighths of every "
                    "flagged run-up token in the split while being a fifth of the "
                    "degenerate answers. Right, coverage inside the loop: code is "
                    "the **lowest** domain in nine of the twelve, the exceptions "
                    "being the three candidates read at layer 4, and the whole "
                    "spread is under a factor of two. So the domain whose loops "
                    "are easiest to see coming is the one whose loops are hardest "
                    "to see once they have arrived, which no single notion of a "
                    "domain being easy accounts for._\n\n"
                    "_Domains are ordered left to right by the median position at "
                    "which their loops start. The left panel rises with that "
                    "order, which is what the next figure exists to rule in or "
                    "out as the cause. `aime_2025` has a single degenerate "
                    "answer and is left out of both panels; it is still "
                    "reported in the table above._"
                ),
            ]
        )

    domain_profile_figure(domain_budget.value)
    return


@app.cell
def _(mo):
    mo.md("""
    A probe's score climbs with position regardless of what the answer is doing,
    and the domains do not put their loops in the same place: code answers begin
    looping around token 1434 on median, instruction-following ones around token
    326. So the ordering above is exactly what a pure position effect would
    produce, and separating the two needs a null.

    The null is the healthy answers of the same domain, read at the same
    absolute positions. Those answers never degenerate, so any flagging they
    attract at a given position is what position alone buys there. The figure
    below puts the two side by side: the run-up tokens of degenerate answers,
    against healthy tokens at matched position, one panel per domain.
    """)
    return


@app.cell
def _(DOMAIN_MINIMUM, LEADING_CANDIDATES, Path, all_runs, np, pd):
    DOMAIN_BANDS = [0, 500, 1000, 1500, 4096]

    def domain_position_null(label, budget):
        """Flag rate against absolute position, for run-up and for healthy tokens.

        Both populations are read from one seed's stored scores at the threshold
        that spends the budget on that seed. A run-up token is any of the 256
        before the loop; a healthy token is any token of an answer that finished
        on its own.
        """
        from degeneration_probe.evaluation.protocol import choose_thresholds

        candidate = next(c for c in LEADING_CANDIDATES if c["label"] == label)
        if all_runs.empty:
            return pd.DataFrame()
        matches = all_runs[
            (all_runs["group"] == candidate["group"])
            & (all_runs["status"] == "finished")
        ].sort_values("seed")
        scores = None
        for run in matches.itertuples():
            path = (
                Path(run.run_dir)
                / "layers"
                / f"layer_{candidate['depth']:02d}"
                / "scores"
                / "val.parquet"
            )
            if path.is_file():
                scores = pd.read_parquet(path)
                break
        if scores is None:
            return pd.DataFrame()

        tau = choose_thresholds(scores, [budget], persistence=1)[0].tau
        bands = np.asarray(DOMAIN_BANDS)
        counts = {}

        def accumulate(domain, population, values, start):
            hits, total = counts.setdefault((domain, population), (np.zeros(len(bands) - 1), np.zeros(len(bands) - 1)))
            index = np.digitize(np.arange(start, start + values.size), bands) - 1
            for band in range(len(bands) - 1):
                chosen = index == band
                total[band] += chosen.sum()
                hits[band] += (values[chosen] >= tau).sum()

        for row in scores.itertuples():
            values = np.asarray(row.scores, dtype=np.float64)
            if row.is_positive:
                onset = int(row.onset_position)
                start = max(0, onset - 256)
                if onset > start:
                    accumulate(row.domain, "run-up of a degenerate answer", values[start:onset], start)
            elif row.stop_reason == "eos":
                accumulate(row.domain, "healthy answer", values, 0)

        big_enough = {
            domain
            for domain, size in scores[scores["is_positive"]]
            .groupby("domain")
            .size()
            .items()
            if size >= DOMAIN_MINIMUM
        }
        rows = []
        for (domain, population), (hits, total) in counts.items():
            if domain not in big_enough:
                continue
            for band in range(len(bands) - 1):
                if not total[band]:
                    continue
                rows.append(
                    {
                        "domain": domain,
                        "population": population,
                        "band": f"{bands[band]}-{bands[band + 1]}",
                        "tokens": int(total[band]),
                        "flag rate": hits[band] / total[band],
                    }
                )
        return pd.DataFrame(rows)

    return DOMAIN_BANDS, domain_position_null


@app.cell
def _(
    DOMAIN_BANDS,
    FIGURES,
    INK_MUTED,
    SERIES,
    domain_budget,
    domain_pick,
    domain_position_null,
    missing,
    mo,
    plt,
    save,
    tidy,
):
    def domain_position_figure(label, budget):
        frame = domain_position_null(label, budget)
        if frame.empty:
            return missing("This candidate has no stored scores on `val` yet.")
        order = [
            f"{DOMAIN_BANDS[i]}-{DOMAIN_BANDS[i + 1]}"
            for i in range(len(DOMAIN_BANDS) - 1)
        ]
        domains = sorted(frame["domain"].unique())
        fig, axes = plt.subplots(
            1, len(domains), figsize=(2.5 * len(domains), 3.4), squeeze=False, sharey=True
        )
        for axis, domain in zip(axes[0], domains):
            block = frame[frame["domain"] == domain]
            for colour, population in zip(SERIES, sorted(block["population"].unique())):
                line = block[block["population"] == population].set_index("band")
                line = line.reindex(order).dropna(subset=["flag rate"])
                if line.empty:
                    continue
                axis.plot(
                    range(len(line)),
                    line["flag rate"].to_numpy(),
                    "o-",
                    color=colour,
                    markersize=4,
                    linewidth=1.4,
                    label=population,
                )
                axis.set_xticks(range(len(line)))
                axis.set_xticklabels(line.index, rotation=40, ha="right", fontsize=7)
            axis.set_yscale("symlog", linthresh=1e-4)
            axis.set_title(domain, fontsize=8, color=INK_MUTED)
            tidy(axis, xgrid=False)
        axes[0][0].set_ylabel("share of tokens flagged")
        seen, handles, names = set(), [], []
        for axis in axes[0]:
            for handle, name in zip(*axis.get_legend_handles_labels()):
                if name not in seen:
                    seen.add(name)
                    handles.append(handle)
                    names.append(name)
        fig.legend(
            handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=2
        )
        fig.suptitle(
            f"{label}: run-up against healthy, at matched position "
            f"({budget:.0%} budget)",
            x=0.005,
            ha="left",
            fontsize=10,
            color=INK_MUTED,
        )
        fig.tight_layout(rect=(0, 0.1, 1, 0.92))
        return mo.vstack(
            [
                save(fig, "domain_position_null", FIGURES),
                mo.md(
                    "_Tokens are grouped by their **absolute** position in the "
                    "answer, so the two lines in a panel are compared at the same "
                    "depth into a generation and position cannot be what "
                    "separates them. The vertical axis is symmetric-log, since "
                    "the rates span four orders of magnitude and a linear axis "
                    "shows only the top band._\n\n"
                    "_The gap between the lines, not the height of either, is the "
                    "quantity of interest. In `llama_nemotron` the run-up runs "
                    "two to three orders of magnitude above its own healthy null "
                    "at every band, so its advantage survives the control "
                    "outright. In the mathematics domains the gap is a factor of "
                    "a few at best, and it closes or reverses outright as "
                    "position grows. In `if_sft_data_verified` the healthy line "
                    "crosses above the run-up line entirely, so past roughly a "
                    "thousand tokens a healthy answer of that domain is more "
                    "likely to be flagged than an answer about to break. The domain ordering in the previous figure is "
                    "therefore not a repackaging of the position effect._"
                ),
            ]
        )

    domain_position_figure(domain_pick.value, domain_budget.value)
    return


@app.cell
def _(mo):
    mo.md("""
    The false alarms split by domain too, and in the opposite direction. Length
    is the confound to remove here rather than position, since the budget is
    spent per answer and a longer answer offers more chances to cross the
    threshold. Healthy answers are therefore grouped by their own length before
    the domains are compared.
    """)
    return


@app.cell
def _(LEADING_CANDIDATES, domain_budget, domain_rollouts, missing, mo, pd):
    def domain_false_alarms(budget):
        """Where the false alarms come from, holding answer length fixed."""
        frames = []
        for candidate in LEADING_CANDIDATES:
            table = domain_rollouts(candidate["label"], budget)
            if table.empty:
                continue
            frames.append(table[~table["is_positive"]])
        if not frames:
            return missing("No candidate has been scored on `val` yet.")
        healthy = pd.concat(frames, ignore_index=True)
        healthy["length"] = pd.cut(
            healthy["num_tokens"],
            [0, 500, 1000, 1500, 4096],
            right=False,
            labels=["under 500", "500-1k", "1k-1.5k", "over 1.5k"],
        )
        rates = healthy.pivot_table(
            index="domain",
            columns="length",
            values="fired",
            aggfunc="mean",
            observed=True,
        ).round(4)
        rates.insert(
            0,
            "share of answers over 1k tokens",
            healthy.groupby("domain")["num_tokens"]
            .apply(lambda lengths: (lengths > 1000).mean())
            .round(3),
        )
        return mo.vstack(
            [
                mo.md(
                    "Share of **healthy** answers raising a false alarm, by "
                    "domain and by their own length, pooled over all twelve "
                    f"candidates and their seeds at a **{budget:.0%} budget**. "
                    "The first column is how much of each domain reaches the "
                    "lengths where false alarms happen at all, so a domain with "
                    "a high rate in the last column and a small share there "
                    "contributes fewer alarms than the rate alone suggests."
                ),
                mo.ui.table(rates.reset_index(), selection=None),
                mo.md(
                    "Length explains most of when a false alarm happens and "
                    "little of where. Within the longest band the mathematics "
                    "domains fire several times more often than `llama_nemotron`, "
                    "which is also the domain with by far the largest share of "
                    "long healthy answers and therefore the most opportunities to "
                    "fire. `if_sft_data_verified` is the one domain that fires on "
                    "short answers at all, which is a mode length cannot account "
                    "for and which matches the repetitive output its prompts "
                    "often ask for."
                ),
            ]
        )

    domain_false_alarms(domain_budget.value)
    return


@app.cell
def _(mo):
    mo.md("""
    **What this changes.** The pooled warning coverage is an average over five
    populations that behave in opposite ways, and it describes none of them.
    Code carries most of the flagged run-up tokens in the whole split while
    supplying the fewest false alarms per long answer; the mathematics domains
    supply most of the false alarms and almost none of the warning; and
    instruction-following has a false-alarm mode of its own that does not need a
    long answer to appear. A single threshold is a compromise between these, and
    every pooled figure elsewhere in this notebook inherits that compromise.

    Read together with the sibling test further down, the same caution applies
    twice. A warning coverage figure is partly a statement about the trajectory
    and partly a statement about the answer's domain, its length, and how far in
    the reader already is. The share attributable to the trajectory alone is
    smaller than the pooled number, and it is not evenly distributed.

    Two of the three per-domain readings sit on small counts: twenty-three
    degenerate code answers and one from `aime_2025`. The claim these support is
    the ordering and its consistency across twelve candidates, not the value in
    any one cell. `codeforces` is held out and is also code, so a scoring pass
    over the held-out split is the natural test of whether this is a property of
    code or of `llama_nemotron`.
    """)
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
def _(replay_controls):
    s3_replay_controls = replay_controls("S3")
    s3_replay_controls
    return (s3_replay_controls,)


@app.cell
def _(replay_scorecard, s3_replay_controls):
    replay_scorecard(
        "S3",
        s3_replay_controls.value["step"],
        s3_replay_controls.value["depth"],
    )
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
    ### Aside: is this probe reading the trajectory, or the prompt?

    Every prompt in this corpus has ten rollouts. Some finish clean every time;
    others produce a mix, at least one degenerate rollout alongside otherwise
    healthy siblings. A probe reading imminence, "this trajectory is about to
    break," should score a healthy sibling of a loop-prone prompt the same as a
    healthy rollout from a prompt that never loops at all, because nothing went
    wrong in either one. A probe that has instead picked up on which *prompts*
    tend to loop, independent of what a given trajectory is doing, will score
    the loop-prone prompt's healthy siblings higher.

    This only works because the corpus keeps ten rollouts per prompt: without a
    finished sibling sitting next to a degenerate one there is nothing to
    compare a healthy trajectory's score against.

    The two probes below are a frozen/adapted pair at depth 15, both
    `all_tokens`/`hard1024`, seed 42 — the frozen head is layer 15 of the S1
    ladder run, the adapted one is S3's LoRA-adapted counterpart at the same
    depth. On validation, 469 healthy rollouts sit beside a degenerate sibling
    and 3,057 come from prompts that never looped. Every healthy rollout's
    score (the same max-over-tokens ranking score used everywhere else in this
    notebook) is split by that grouping, and the two populations are compared
    with a rank-biserial AUC: the probability a random sibling outscores a
    random clean rollout. 0.5 is what a pure imminence detector should read,
    since the two populations differ only in a fact about the prompt the probe
    never sees; anything higher is the probe using that fact anyway.
    """)
    return


@app.cell
def _(BUILD_ROOT, OUTPUTS, pd):
    def prompt_predisposition_labels(split="val"):
        """Which prompts in this split ever produced a degenerate rollout."""
        labels = pd.read_parquet(
            BUILD_ROOT / "onset_labels" / "onset_labels.parquet",
            columns=["prompt_id", "split", "is_positive"],
        )
        labels = labels[labels["split"] == split]
        return labels.groupby("prompt_id")["is_positive"].any()

    PREDISPOSITION_PROBES = {
        "frozen, L15": OUTPUTS
        / "apertus-8b-instruct_L1-31_all_tokens_hard1024_bce_lora-none_s42_7790c264"
        / "20260814T050453" / "layers" / "layer_15" / "scores" / "val.parquet",
        "lora-adapted, L15": OUTPUTS
        / "apertus-8b-instruct_L15_all_tokens_hard1024_bce_lora-all_s42_b9e9df34"
        / "20260816T085625" / "scores" / "val.parquet",
    }
    return PREDISPOSITION_PROBES, prompt_predisposition_labels


@app.cell
def _(PREDISPOSITION_PROBES, pd, prompt_predisposition_labels):
    from degeneration_probe.evaluation.protocol import (
        choose_thresholds as _choose_thresholds,
        rollout_score as _rollout_score,
    )
    from scipy import stats as scipy_stats

    def _healthy_groups(frame, loop_prone):
        """Finished-on-its-own rollouts, split by their prompt's own history.

        Healthy means stop_reason == 'eos' rather than merely is_positive ==
        False, which would also admit the handful of capped rollouts the judge
        could not resolve -- not a clean trajectory to compare against.
        """
        healthy = frame[frame["stop_reason"] == "eos"].copy()
        healthy["prompt_loop_prone"] = healthy["prompt_id"].map(loop_prone).astype(bool)
        return (
            healthy[healthy["prompt_loop_prone"]],
            healthy[~healthy["prompt_loop_prone"]],
        )

    def predisposition_result(path):
        frame = pd.read_parquet(path)
        frame["rollout_score"] = [_rollout_score(s) for s in frame["scores"]]
        siblings, clean = _healthy_groups(frame, prompt_predisposition_labels())

        statistic, p_value = scipy_stats.mannwhitneyu(
            siblings["rollout_score"], clean["rollout_score"], alternative="greater"
        )
        auc = float(statistic / (len(siblings) * len(clean)))

        thresholds = _choose_thresholds(frame, [0.01, 0.05, 0.10], persistence=1)
        rates = pd.DataFrame(
            [
                {
                    "target_negative_fpr": threshold.target_negative_fpr,
                    "sibling_alarm_rate": float(
                        (siblings["rollout_score"] >= threshold.tau).mean()
                    ),
                    "clean_alarm_rate": float(
                        (clean["rollout_score"] >= threshold.tau).mean()
                    ),
                }
                for threshold in thresholds
            ]
        )
        rates["ratio"] = rates["sibling_alarm_rate"] / rates["clean_alarm_rate"]
        return {
            "siblings": siblings,
            "clean": clean,
            "auc": auc,
            "p_value": float(p_value),
            "rates": rates,
        }

    PREDISPOSITION_RESULTS = {
        label: predisposition_result(path) for label, path in PREDISPOSITION_PROBES.items()
    }
    return (PREDISPOSITION_RESULTS,)


@app.cell
def _(
    FIGURES,
    INK_SOFT,
    PREDISPOSITION_RESULTS,
    SERIES,
    mo,
    np,
    plt,
    save,
    tidy,
):
    def predisposition_figure():
        """The healthy-score distribution, sibling-of-loop-prone vs clean."""
        labels = list(PREDISPOSITION_RESULTS)
        fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 3.6), sharey=True)
        if len(labels) == 1:
            axes = [axes]
        for axis, label, colour in zip(axes, labels, SERIES):
            result = PREDISPOSITION_RESULTS[label]
            for key, name, style in (
                ("clean", "clean prompt", {"linestyle": "-"}),
                ("siblings", "sibling of a loop-prone prompt", {"linestyle": (0, (3, 2))}),
            ):
                values = np.sort(result[key]["rollout_score"].to_numpy())
                fractions = np.arange(1, len(values) + 1) / len(values)
                axis.plot(values, fractions, color=colour, label=name, **style)
            axis.set_title(f"{label}  (AUC={result['auc']:.3f})", color=INK_SOFT)
            axis.set_xlabel("rollout score (max over tokens)")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            tidy(axis, xgrid=False)
            axis.legend(loc="lower right")
        axes[0].set_ylabel("cumulative share of healthy rollouts")
        fig.suptitle(
            "Healthy-rollout score, by whether the prompt has a degenerate sibling",
            x=0.005, ha="left", fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return mo.vstack(
            [
                save(fig, "prompt_predisposition_ecdf", FIGURES),
                mo.md(
                    "_Cumulative distribution of the healthy-rollout score, split by "
                    "whether the rollout's own prompt ever produced a degenerate "
                    "sibling. A pure imminence detector would draw the two lines on "
                    "top of each other; the dashed line sitting to the right of the "
                    "solid one is the probe reading the prompt rather than only the "
                    "trajectory in front of it._"
                ),
            ]
        )

    predisposition_figure()
    return


@app.cell
def _(PREDISPOSITION_RESULTS, mo, pd):
    def predisposition_summary():
        rows = []
        for label, result in PREDISPOSITION_RESULTS.items():
            row = {
                "probe": label,
                "healthy siblings": len(result["siblings"]),
                "healthy, clean prompt": len(result["clean"]),
                "AUC (sibling > clean)": round(result["auc"], 3),
                "p-value": f"{result['p_value']:.1e}",
            }
            for _, rate in result["rates"].iterrows():
                row[f"alarm ratio @{rate['target_negative_fpr']:.0%}"] = round(rate["ratio"], 2)
            rows.append(row)
        return mo.vstack(
            [
                mo.ui.table(pd.DataFrame(rows), selection=None),
                mo.md(
                    "Both probes read some of the prompt's own predisposition into a "
                    "trajectory that never showed anything: a healthy rollout is "
                    "roughly two to three times more likely to trip the alarm if it "
                    "shares a prompt with a rollout that degenerated, at every budget "
                    "tested. Letting the adapter reshape the representation, S3's own "
                    "question, does not fix this -- the LoRA-adapted probe's AUC is "
                    "not lower than the frozen one's, if anything slightly higher. "
                    "This does not by itself prove a probe cannot also read "
                    "imminence; it shows that whatever it reads is not cleanly "
                    "separated from prompt-level difficulty, worth keeping in mind "
                    "whenever a warning-coverage number elsewhere in this notebook is "
                    "read as \"the probe saw the approach\" rather than \"the probe "
                    "suspected the prompt.\""
                ),
            ]
        )

    predisposition_summary()
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
    Apertus, just pointed at cached activations from a different model. Four
    targets, in two pairs: `meta-llama/Llama-3.1-8B-Instruct` and
    `mistralai/Mistral-7B-Instruct-v0.1` are a different architecture family
    entirely; the other two are Apertus 1.5 variants (`capfilter-linear-it8816`,
    `sft256k-4200`), the same architecture the checkpoints were trained on but a
    different set of weights. All four share Apertus's hidden size of 4096 and
    its 32 layers, so a saved linear head is shape-compatible everywhere; the
    two pairs exist to separate two different questions the shape-compatibility
    alone cannot answer: does the direction the probe found exist in a wholly
    different model's activations at all, and separately, does it survive a
    change of weights within the same architecture the head was picked on.

    The twelve leading candidates are what gets transferred, three seeds each,
    every one at the depth it was scored at. Only frozen checkpoints are ever
    scored this way: a LoRA adapter is fitted to Apertus's own decoder weights
    and has no meaning against another model's activations. Part 4 below asks
    the adapter's own version of this question instead: not whether its
    activations transfer, but whether the adapter itself, transplanted onto a
    different checkpoint's weights, still does its job.

    Everything here reads the validation split. The transferred side has the two
    test splits on disk as well, but the Apertus-side numbers each result is
    compared against exist only on validation, so a comparison on anything else
    would have nothing to sit beside.

    Ground truth for the target models comes from the same LLM judge as every
    other split in this notebook, so a row below is only as trustworthy as that
    judge run on that model's answers.
    """)
    return


@app.cell
def _(LEADING_CANDIDATES, Path, all_runs, pd):
    # The four models every candidate has been scored against. Discovery is by
    # candidate rather than by walking the output tree, so a half-finished sweep
    # against some other model cannot quietly add itself to the comparison.
    CROSS_MODEL_TARGETS = {
        "llama3p1-8b-instruct": "llama 3.1 8b",
        "mistral-7b-instruct-v0p1": "mistral 7b",
        "apertus1p5-capfilter-linear-it8816": "apertus 1.5 capfilter",
        "apertus1p5-sft256k-4200": "apertus 1.5 sft256k",
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
        f"{len(LEADING_CANDIDATES) * 3 * 4} expected: twelve candidates, three "
        "seeds each, four target models."
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
        candidates times four target models plus twelve native references is
        sixty lines, and the comparison that matters is within a candidate,
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
    **In-pattern coverage on Llama and Mistral below is genuinely zero, and it
    is not a plotting fault.** It is worth saying why, because zero everywhere
    usually means a broken axis. On the two Apertus 1.5 targets, by contrast,
    it is nowhere near zero: 0.65 median in-pattern coverage on capfilter,
    0.58 on sft256k, against 0.000 on both Llama and Mistral, all at the same
    5% false-alarm budget on the same twelve candidates. So the shape
    compatibility all four targets share is not what decides this: the two
    that share Apertus's own weights transfer; the two that only share its
    hidden size and layer count do not.

    Transferred to Llama or Mistral, the head stops separating anything: on
    Llama every healthy answer peaks somewhere between 0.58 and 0.97, and the
    tokens inside the loop reach at most 0.94. The threshold that spends even a
    1% false-alarm budget therefore sits at about 0.95, above the highest score
    any in-loop token achieves, so nothing inside the loop is flagged and the
    rate is exactly 0. On Apertus the same head puts in-loop tokens near 0.93 on
    average while healthy answers spread all the way down to 0.14, and the same
    1% threshold still admits two thirds of them.

    So the head does not produce a weaker version of the same signal on a
    different architecture. It produces a compressed, uniformly high score
    that carries almost no ordering, which is also why rollout recall sits
    near zero at small budgets on those two targets, but not on the other two.
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
    # Part 4. Does the adapter itself transplant?

    Part 3 asked whether the *direction* a frozen head found survives on
    another model's activations. A LoRA-adapted checkpoint has no direction to
    export that way: the adapter reshapes Apertus's own decoder weights, and
    the probe head is fit to the reshaped result, so the pair only means
    anything read off Apertus's own layers. Scored against a different
    architecture's activations, as Part 3's intro already noted, this is not
    weak transfer, it is a category error, and it is not attempted anywhere in
    this part.

    What is attempted instead: load a *different* checkpoint's weights, one
    that is the same architecture as Apertus but not the same weights, apply
    the source run's adapter and head to it exactly as `setup_probe` applies
    them during training, and score it live, one real forward pass per
    rollout, no cached activations anywhere in the path. The two Apertus 1.5
    variants are the only targets this is possible for at all, since they are
    the only other checkpoints sharing Apertus's architecture. The question
    this answers: does the adapter's reshaping of the residual stream capture
    something about degeneration that generalises across weights, or is it
    fit to the exact checkpoint it was trained on and nothing else.

    The candidate set is close to Part 3's twelve but not identical: S1 #3
    (`frontier_hard_negative128`) was never trained under LoRA, and S2c, a
    regression head on `repetition_score` that Part 2 ruled out early and so
    never appears there, was. Twelve configurations, three seeds each, all at
    the depth each was originally scored at.

    Everything here reads the validation split, same as Part 3, and for the
    same reason: the native side of the comparison, this exact checkpoint
    scored on Apertus itself under the same adapter weights, only exists on
    validation, saved once per checkpoint under `sweep/<checkpoint>/` as part
    of picking it in the first place.
    """)
    return


@app.cell
def _():
    # The transplant sweep's own checkpoint list, read off training's own
    # output directories the same way Part 2 reads LEADING_CANDIDATES: not
    # re-derived here, only named and pointed at. Depth, seed and checkpoint
    # are each confirmed against the run directories themselves.
    LORA_TRANSPLANT_CANDIDATES = [
        {"stage": "S1", "label": "S1 #1", "summary": "rollout_balanced, W=128", "depth": 15, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L15_rollout_balanced128_hard_bce_lora-all_s42_ba87b2e1/20260816T085621", "checkpoint": "checkpoint-700"},
        {"stage": "S1", "label": "S1 #1", "summary": "rollout_balanced, W=128", "depth": 15, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L15_rollout_balanced128_hard_bce_lora-all_s43_70ce1945/20260816T085625", "checkpoint": "checkpoint-1700"},
        {"stage": "S1", "label": "S1 #1", "summary": "rollout_balanced, W=128", "depth": 15, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L15_rollout_balanced128_hard_bce_lora-all_s44_2cbd1630/20260816T085625", "checkpoint": "checkpoint-500"},
        {"stage": "S1", "label": "S1 #2", "summary": "frontier_window, W=256, centered", "depth": 15, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L15_frontier256_hard_bce_lora-all_s42_9730324c/20260816T085622", "checkpoint": "checkpoint-300"},
        {"stage": "S1", "label": "S1 #2", "summary": "frontier_window, W=256, centered", "depth": 15, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L15_frontier256_hard_bce_lora-all_s43_184d454f/20260816T085622", "checkpoint": "checkpoint-850"},
        {"stage": "S1", "label": "S1 #2", "summary": "frontier_window, W=256, centered", "depth": 15, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L15_frontier256_hard_bce_lora-all_s44_f93b2881/20260816T085620", "checkpoint": "checkpoint-1150"},
        {"stage": "S2a", "label": "S2a #1", "summary": "frontier_window, W=512, centered, horizon=256", "depth": 15, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_hard256_bce_lora-all_s42_16f11891/20260816T085615", "checkpoint": "checkpoint-1100"},
        {"stage": "S2a", "label": "S2a #1", "summary": "frontier_window, W=512, centered, horizon=256", "depth": 15, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_hard256_bce_lora-all_s43_10cb63a6/20260816T085621", "checkpoint": "checkpoint-1150"},
        {"stage": "S2a", "label": "S2a #1", "summary": "frontier_window, W=512, centered, horizon=256", "depth": 15, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_hard256_bce_lora-all_s44_defb4925/20260816T085629", "checkpoint": "checkpoint-1150"},
        {"stage": "S2a", "label": "S2a #2", "summary": "all_tokens, horizon=1024", "depth": 15, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L15_all_tokens_hard1024_bce_lora-all_s42_b9e9df34/20260816T085625", "checkpoint": "checkpoint-1600"},
        {"stage": "S2a", "label": "S2a #2", "summary": "all_tokens, horizon=1024", "depth": 15, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L15_all_tokens_hard1024_bce_lora-all_s43_1877dd23/20260816T085623", "checkpoint": "checkpoint-1700"},
        {"stage": "S2a", "label": "S2a #2", "summary": "all_tokens, horizon=1024", "depth": 15, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L15_all_tokens_hard1024_bce_lora-all_s44_6c4843a3/20260816T085627", "checkpoint": "checkpoint-1600"},
        {"stage": "S2a", "label": "S2a #3", "summary": "frontier_window, W=512, trailing, horizon=512", "depth": 15, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_hard512_bce_lora-all_s42_3983be82/20260816T085631", "checkpoint": "checkpoint-700"},
        {"stage": "S2a", "label": "S2a #3", "summary": "frontier_window, W=512, trailing, horizon=512", "depth": 15, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_hard512_bce_lora-all_s43_b9b42c7f/20260816T085625", "checkpoint": "checkpoint-1250"},
        {"stage": "S2a", "label": "S2a #3", "summary": "frontier_window, W=512, trailing, horizon=512", "depth": 15, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_hard512_bce_lora-all_s44_7c3d8248/20260816T085633", "checkpoint": "checkpoint-1150"},
        {"stage": "S2b", "label": "S2b #1", "summary": "soft label, exponential decay/256", "depth": 15, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_soft_bce_lora-all_s42_437c96ea/20260816T085628", "checkpoint": "checkpoint-700"},
        {"stage": "S2b", "label": "S2b #1", "summary": "soft label, exponential decay/256", "depth": 15, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_soft_bce_lora-all_s43_1d6b2028/20260816T085626", "checkpoint": "checkpoint-1250"},
        {"stage": "S2b", "label": "S2b #1", "summary": "soft label, exponential decay/256", "depth": 15, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_soft_bce_lora-all_s44_f805d430/20260816T085721", "checkpoint": "checkpoint-750"},
        {"stage": "S2b", "label": "S2b #2", "summary": "soft label, linear decay/256", "depth": 15, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_soft_bce_lora-all_s42_58f70906/20260816T085631", "checkpoint": "checkpoint-700"},
        {"stage": "S2b", "label": "S2b #2", "summary": "soft label, linear decay/256", "depth": 15, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_soft_bce_lora-all_s43_3d8a070f/20260816T085629", "checkpoint": "checkpoint-450"},
        {"stage": "S2b", "label": "S2b #2", "summary": "soft label, linear decay/256", "depth": 15, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_soft_bce_lora-all_s44_c40acb2c/20260816T085634", "checkpoint": "checkpoint-750"},
        {"stage": "S2b", "label": "S2b #3", "summary": "soft label, exponential decay/128", "depth": 15, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_soft_bce_lora-all_s42_9ef8b036/20260816T085633", "checkpoint": "checkpoint-1100"},
        {"stage": "S2b", "label": "S2b #3", "summary": "soft label, exponential decay/128", "depth": 15, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_soft_bce_lora-all_s43_e91a9911/20260816T085626", "checkpoint": "checkpoint-450"},
        {"stage": "S2b", "label": "S2b #3", "summary": "soft label, exponential decay/128", "depth": 15, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L15_frontier512_soft_bce_lora-all_s44_f1ba47d3/20260816T085629", "checkpoint": "checkpoint-750"},
        {"stage": "S2c", "label": "S2c", "summary": "regression on repetition_score, MSE", "depth": 12, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L12_frontier128_repetition_score_mse_lora-all_s42_d1708e38/20260816T085632", "checkpoint": "checkpoint-1450"},
        {"stage": "S2c", "label": "S2c", "summary": "regression on repetition_score, MSE", "depth": 12, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L12_frontier128_repetition_score_mse_lora-all_s43_d93bc55e/20260816T085634", "checkpoint": "checkpoint-1500"},
        {"stage": "S2c", "label": "S2c", "summary": "regression on repetition_score, MSE", "depth": 12, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L12_frontier128_repetition_score_mse_lora-all_s44_fcb9756d/20260816T085631", "checkpoint": "checkpoint-1100"},
        {"stage": "S2d", "label": "S2d #1", "summary": "base recipe, pos_weight=on", "depth": 4, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L4_frontier128_hard_bce_lora-all_s42_2f0b0bd2/20260816T085632", "checkpoint": "checkpoint-1200"},
        {"stage": "S2d", "label": "S2d #1", "summary": "base recipe, pos_weight=on", "depth": 4, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L4_frontier128_hard_bce_lora-all_s43_26446f82/20260816T085639", "checkpoint": "checkpoint-1100"},
        {"stage": "S2d", "label": "S2d #1", "summary": "base recipe, pos_weight=on", "depth": 4, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L4_frontier128_hard_bce_lora-all_s44_77556c4d/20260816T085636", "checkpoint": "checkpoint-750"},
        {"stage": "S2d", "label": "S2d #2", "summary": "base recipe, pos_weight=off", "depth": 4, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L4_frontier128_hard_bce_lora-all_s42_6784e5dc/20260816T085633", "checkpoint": "checkpoint-550"},
        {"stage": "S2d", "label": "S2d #2", "summary": "base recipe, pos_weight=off", "depth": 4, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L4_frontier128_hard_bce_lora-all_s43_477c135c/20260816T085637", "checkpoint": "checkpoint-550"},
        {"stage": "S2d", "label": "S2d #2", "summary": "base recipe, pos_weight=off", "depth": 4, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L4_frontier128_hard_bce_lora-all_s44_686bb168/20260816T085636", "checkpoint": "checkpoint-500"},
        {"stage": "S2d", "label": "S2d #3", "summary": "base recipe, pos_weight=on, positive_fraction=0.5", "depth": 4, "seed": 42, "run_dir": "outputs/apertus-8b-instruct_L4_frontier128_hard_bce_lora-all_s42_19312f07/20260816T085636", "checkpoint": "checkpoint-1200"},
        {"stage": "S2d", "label": "S2d #3", "summary": "base recipe, pos_weight=on, positive_fraction=0.5", "depth": 4, "seed": 43, "run_dir": "outputs/apertus-8b-instruct_L4_frontier128_hard_bce_lora-all_s43_65b4300f/20260816T085628", "checkpoint": "checkpoint-1100"},
        {"stage": "S2d", "label": "S2d #3", "summary": "base recipe, pos_weight=on, positive_fraction=0.5", "depth": 4, "seed": 44, "run_dir": "outputs/apertus-8b-instruct_L4_frontier128_hard_bce_lora-all_s44_7a0c77d6/20260816T085639", "checkpoint": "checkpoint-750"},
    ]
    return (LORA_TRANSPLANT_CANDIDATES,)


@app.cell
def _(LORA_TRANSPLANT_CANDIDATES, Path, pd):
    # The two possible targets: the only other checkpoints sharing Apertus's
    # architecture. Discovery is by candidate, same reasoning as Part 3's
    # cross_model_index -- a half-finished sweep cannot quietly add itself.
    LORA_TARGETS = {
        "apertus1p5-capfilter-linear-it8816": "apertus 1.5 capfilter",
        "apertus1p5-sft256k-4200": "apertus 1.5 sft256k",
    }
    LORA_SPLIT = "val"

    def lora_transplant_index():
        """One row per (candidate, seed, target checkpoint) that has been scored."""
        rows = []
        for candidate in LORA_TRANSPLANT_CANDIDATES:
            root = Path(candidate["run_dir"])
            native = root / "sweep" / candidate["checkpoint"]
            for target, target_label in LORA_TARGETS.items():
                scoped = root / "lora_transplant" / target
                evaluation = scoped / "evaluation" / LORA_SPLIT
                if not (evaluation / "view_a_detection.csv").is_file():
                    continue
                rows.append(
                    {
                        "stage": candidate["stage"],
                        "label": candidate["label"],
                        "recipe": candidate["summary"],
                        "depth": candidate["depth"],
                        "target": target_label,
                        "seed": candidate["seed"],
                        "evaluation": str(evaluation),
                        "scores": str(scoped / "scores" / f"{LORA_SPLIT}.parquet"),
                        "native_scores": str(native / "scores" / f"{LORA_SPLIT}.parquet"),
                    }
                )
        return pd.DataFrame(rows)

    LORA_TRANSPLANT = lora_transplant_index()
    return (LORA_TRANSPLANT,)


@app.cell
def _(LORA_TRANSPLANT, LORA_TRANSPLANT_CANDIDATES, missing, mo):
    mo.stop(
        LORA_TRANSPLANT.empty,
        missing(
            "No transplant scores on disk yet. They land under "
            "`<run_dir>/lora_transplant/<model>/evaluation/<split>/` once "
            "`scripts/evaluate_lora_transplant.py` and "
            "`scripts/evaluate_scores.py` have run for a checkpoint."
        ),
    )
    mo.md(
        f"**{len(LORA_TRANSPLANT)} scored transplants found**, out of "
        f"{len(LORA_TRANSPLANT_CANDIDATES) * 2} expected: thirty-six checkpoints, "
        "two target checkpoints each."
    )
    return


@app.cell
def _(budget_control):
    lora_transplant_table_budget = budget_control()
    lora_transplant_table_budget
    return (lora_transplant_table_budget,)


@app.cell
def _(LORA_TRANSPLANT, Path, lora_transplant_table_budget, mo, pd):
    # Same view set as Part 3's transfer_table, kept local to this cell under
    # underscore-prefixed names so the two sections can define their own
    # copies without marimo treating them as the same variable twice.
    _LORA_VIEWS = [
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
    _LORA_PERSISTENCE = [
        ("alarm length, degenerate", "positive"),
        ("alarm length, healthy", "negative"),
    ]
    _LORA_ORDER = [name for name, *_ in _LORA_VIEWS] + [
        name for name, _ in _LORA_PERSISTENCE
    ]

    def _lora_transfer_records(budget):
        records = []
        for record in LORA_TRANSPLANT.itertuples():
            evaluation = Path(record.evaluation)
            for metric, view_file, column in _LORA_VIEWS:
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
            for metric, population in _LORA_PERSISTENCE:
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

    def _lora_transfer_table(budget):
        long = _lora_transfer_records(budget)
        if long.empty:
            return mo.md("_Nothing scored at this budget yet._")

        def _summarise(values):
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
            aggfunc=_summarise,
        ).reindex(columns=[m for m in _LORA_ORDER if m in set(long["metric"])])
        table = meta.join(pivoted).reset_index()
        table = table.sort_values(["label", "target"])
        return mo.vstack(
            [
                mo.md(
                    "One row per candidate per target checkpoint: the median "
                    "across its three seeds, with the seed-to-seed range in "
                    f"brackets where it varies, at a **{budget:.0%} false-alarm "
                    "budget** on the target checkpoint's own validation split."
                ),
                mo.ui.table(table, selection=None),
            ]
        )

    _lora_transfer_table(lora_transplant_table_budget.value)
    return


@app.cell
def _(LORA_TRANSPLANT, np, pd):
    from degeneration_probe.evaluation.protocol import (
        coverage_window as _lora_coverage_window,
        rollout_score as _lora_rollout_score,
        threshold_for_budget as _lora_threshold_for_budget,
    )

    _LORA_BUDGET_GRID = np.geomspace(0.001, 0.30, 48)
    _LORA_SWEEP_CACHE = {}

    def _lora_sweep_one(path):
        if path in _LORA_SWEEP_CACHE:
            return _LORA_SWEEP_CACHE[path]
        frame = pd.read_parquet(path)
        positives = frame[frame["is_positive"].astype(bool)]
        negatives = frame[~frame["is_positive"].astype(bool)]
        if positives.empty or negatives.empty:
            _LORA_SWEEP_CACHE[path] = pd.DataFrame()
            return _LORA_SWEEP_CACHE[path]
        negative_peaks = np.array(
            [_lora_rollout_score(s) for s in negatives["scores"]], dtype=np.float64
        )
        positive_peaks = np.array(
            [_lora_rollout_score(s) for s in positives["scores"]], dtype=np.float64
        )
        in_pattern = np.sort(
            np.concatenate(
                [
                    np.asarray(scores, dtype=np.float64)[
                        _lora_coverage_window(len(scores), int(onset), None)
                    ]
                    for scores, onset in zip(
                        positives["scores"], positives["onset_position"]
                    )
                ]
            )
        )
        rows = []
        for budget in _LORA_BUDGET_GRID:
            tau, _ = _lora_threshold_for_budget(negative_peaks, float(budget))
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
        _LORA_SWEEP_CACHE[path] = pd.DataFrame(rows)
        return _LORA_SWEEP_CACHE[path]

    def _lora_transplant_sweeps():
        frames = []
        for record in LORA_TRANSPLANT.itertuples():
            for source, path in (
                (record.target, record.scores),
                ("apertus (native)", record.native_scores),
            ):
                curve = _lora_sweep_one(path)
                if curve.empty:
                    continue
                piece = curve.copy()
                piece["label"] = record.label
                piece["source"] = source
                piece["seed"] = record.seed
                frames.append(piece)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    LORA_SWEEPS = _lora_transplant_sweeps()
    return (LORA_SWEEPS,)


@app.cell
def _(
    FIGURES,
    INK_SOFT,
    LORA_SWEEPS,
    LORA_TRANSPLANT_CANDIDATES,
    SERIES,
    missing,
    mo,
    plt,
    save,
    tidy,
):
    def lora_budget_sweep_figure(column, title, ylabel, name):
        """One metric against the false-alarm budget, per candidate.

        Same layout as Part 3's budget_sweep_figure: a panel per candidate,
        solid lines for each target checkpoint, a dashed line for the same
        checkpoint's native (in-family) score.
        """
        if LORA_SWEEPS.empty:
            return missing("No transplant has been swept yet.")
        labels = list(dict.fromkeys(c["label"] for c in LORA_TRANSPLANT_CANDIDATES))
        labels = [label for label in labels if label in set(LORA_SWEEPS["label"])]
        if not labels:
            return missing("No transplant has been swept yet.")
        sources = sorted(s for s in LORA_SWEEPS["source"].unique() if s != "apertus (native)")
        colours = dict(zip(sources, SERIES))

        columns = 4
        rows = -(-len(labels) // columns)
        fig, axes = plt.subplots(
            rows, columns, figsize=(3.0 * columns, 2.5 * rows), squeeze=False,
            sharex=True, sharey=True,
        )
        flat = [axis for row in axes for axis in row]
        for axis, label in zip(flat, labels):
            panel = LORA_SWEEPS[LORA_SWEEPS["label"] == label]
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
                    "Solid lines are the adapter and head transplanted onto a "
                    "sibling checkpoint's weights; the dashed line is the same "
                    "adapter and head on the Apertus checkpoint they were "
                    "actually trained on, read at the same budget. The budget "
                    "axis is swept continuously the same way Part 3's is._"
                ),
            ]
        )

    return (lora_budget_sweep_figure,)


@app.cell
def _(lora_budget_sweep_figure):
    lora_budget_sweep_figure(
        "recall",
        "Detection: does the transplanted adapter still catch a degenerate answer?",
        "recall",
        "lora_transplant_recall_vs_budget",
    )
    return


@app.cell
def _(lora_budget_sweep_figure):
    lora_budget_sweep_figure(
        "in_pattern_recall",
        "Coverage: does it still flag the tokens inside the loop?",
        "in-pattern coverage",
        "lora_transplant_in_pattern_vs_budget",
    )
    return


@app.cell
def _(lora_budget_sweep_figure):
    lora_budget_sweep_figure(
        "precision",
        "Precision: what share of its alarms are on degenerate answers?",
        "precision",
        "lora_transplant_precision_vs_budget",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    **Transplant trades coverage for precision, compared to reading the same
    signal off frozen activations.** Both routes reach similarly high recall
    on both Apertus 1.5 targets, and the ceiling on either route touches 1.0
    for several candidates. Where they part ways: the transplanted adapter's
    best precision at a fixed budget clears 0.85-0.93 on some candidates,
    well above anything the frozen head reaches on these same two targets,
    but its in-pattern coverage and warning lead time both run lower on
    average than the frozen head's. A transplanted adapter fires sharper and
    closer to the actual onset; a frozen head fires on more of the span
    around it, including earlier, weaker signs, at the cost of more false
    alarms along the way. Neither route dominates the other, and which one a
    deployment wants depends on whether a late, confident alarm or an early,
    noisier one is worth more.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Part 5. Reading the same probe under someone else's conventions

    Every number in this notebook is measured one way: a decision per token, a
    threshold solved so that one healthy answer in a hundred raises a false
    alarm, and a population left at its natural rate of degeneration. The
    nearest published work measures differently on all three counts. It pools
    tokens into sentences, accumulates the evidence across them, reads an
    operating point where a quarter to a third of healthy generations may fire,
    and evaluates on a set rebalanced to equal numbers of the two classes.

    Those three are decisions about how a score becomes a decision, not about
    the scorer. So they can be turned on scores that already exist. Holding one
    probe fixed and changing them one at a time gives a ladder from this
    protocol to that one, and the distance between rungs is how much of the
    disagreement between published numbers is a matter of convention.

    Nothing here evaluates anyone else's detector. It is one of our probes read
    under other people's reporting choices, and it says nothing about what their
    model would do on this data.
    """)
    return


@app.cell
def _(OUTPUTS, Path, json):
    import importlib.util as _importlib_util

    # The ladder is computed by scripts/protocol_bridge.py rather than
    # reimplemented here, so the figure below and the command line answer cannot
    # drift apart.
    def _load_bridge():
        path = Path(__file__).resolve().parents[1] / "scripts" / "protocol_bridge.py"
        spec = _importlib_util.spec_from_file_location("protocol_bridge", path)
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    BRIDGE = _load_bridge()

    def bridge_candidates():
        """Scored validation splits that can name the checkpoint that produced them.

        Only directories carrying a provenance record are offered. A scores file
        whose weights cannot be named is fine for a trend and wrong for a table
        that will be quoted, and this is the second kind.
        """
        found = {}
        for scores in OUTPUTS.glob("*/*/sweep/layer_*/checkpoint-*/scores/val.parquet"):
            record = scores.parent.parent / "scoring_provenance.json"
            if not record.is_file():
                continue
            provenance = json.loads(record.read_text())
            # scores/ checkpoint-N/ layer_NN/ sweep/ <attempt>/ <run name>
            run = scores.parents[5].name
            # Drop only the configuration fingerprint. Truncating further would
            # cut the seed suffix and collapse three runs of one recipe into one
            # entry.
            label = (
                f"{run.rsplit('_', 1)[0]} "
                f"L{provenance['layer']} {provenance['checkpoint']}"
            )
            found[label] = scores
        return dict(sorted(found.items()))

    BRIDGE_CANDIDATES = bridge_candidates()
    return BRIDGE, BRIDGE_CANDIDATES


@app.cell
def _(BRIDGE_CANDIDATES, missing, mo):
    mo.stop(
        not BRIDGE_CANDIDATES,
        missing(
            "No scored validation split carries a provenance record yet. They are "
            "written by `scripts/score_rollouts.py` from the commit that added "
            "them onward, and land beside the scores as `scoring_provenance.json`."
        ),
    )
    bridge_pick = mo.ui.dropdown(
        options=BRIDGE_CANDIDATES,
        value=next(iter(BRIDGE_CANDIDATES)),
        label="probe",
    )
    bridge_span = mo.ui.dropdown(
        options={"8": 8, "16": 16, "32": 32, "64": 64, "128": 128},
        value="32",
        label="tokens per pooled span",
    )
    mo.hstack([bridge_pick, bridge_span], justify="start", gap=2)
    return bridge_pick, bridge_span


@app.cell
def _(BRIDGE, bridge_pick, bridge_span, pd):
    def bridge_table(scores_path, span):
        frame = pd.read_parquet(scores_path)
        return BRIDGE.build(frame, span, [0.01, 0.05, 0.10, 0.30], draws=20, seed=42)

    BRIDGE_RESULT = bridge_table(bridge_pick.value, bridge_span.value)
    return (BRIDGE_RESULT,)


@app.cell
def _(BRIDGE_RESULT, mo):
    def bridge_tables():
        columns = [
            "step", "recall", "precision", "realized_fpr", "in_pattern",
            "warning_256", "median_offset", "fired_before", "never_fired",
        ]
        blocks = []
        for name, block in BRIDGE_RESULT.groupby("table", sort=False):
            blocks.append(mo.md(f"**{name}**"))
            blocks.append(mo.ui.table(block[columns].round(4), selection=None))
        blocks.append(
            mo.md(
                "_The first table walks from this protocol to the other one, "
                "changing one convention per row and keeping the earlier changes. "
                "The second changes one at a time from the top rung, which is what "
                "says how large each term is on its own. **No weight changes in "
                "either table.** `realized_fpr` is the share of healthy answers "
                "that fired, so it reads back the budget the row was solved for._\n\n"
                "_A pooled decision is carried from the **last** token of its span "
                "rather than the first. Crediting it from the first would hand a "
                "span detector up to a span of lead time it never had, which is "
                "the quantity being measured. A rollout shorter than one span "
                "therefore never fires, which is what a detector that must see a "
                "whole span before deciding actually does._"
            )
        )
        return mo.vstack(blocks)

    bridge_tables()
    return


@app.cell
def _(BRIDGE_RESULT, FIGURES, INK_MUTED, SERIES, mo, plt, save, tidy):
    def bridge_ladder_figure():
        """The ladder as two panels: how much is flagged, and when."""
        ladder = BRIDGE_RESULT[BRIDGE_RESULT["table"] == "ladder"]
        labels = [step.split(". ", 1)[-1] for step in ladder["step"]]
        positions = range(len(ladder))

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
        axes[0].plot(
            positions, ladder["warning_256"], "o-", color=SERIES[0], linewidth=1.8
        )
        axes[0].set_ylabel("coverage of the 256 before the loop")
        axes[0].set_ylim(bottom=0)

        axes[1].plot(
            positions, ladder["median_offset"], "o-", color=SERIES[1], linewidth=1.8
        )
        axes[1].axhline(0, color=INK_MUTED, linewidth=1, zorder=0)
        axes[1].set_ylabel("median first alarm, tokens from the loop")

        for axis in axes:
            axis.set_xticks(list(positions))
            axis.set_xticklabels(labels, rotation=28, ha="right", fontsize=7.5)
            tidy(axis, xgrid=False)

        fig.suptitle(
            "One probe, five reporting conventions", x=0.005, ha="left",
            fontsize=10, color=INK_MUTED,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return mo.vstack(
            [
                save(fig, "protocol_bridge_ladder", FIGURES),
                mo.md(
                    "_Left, the share of the run-up that gets flagged. Right, where "
                    "the first alarm lands, with zero marking the loop's own start "
                    "and negative meaning the alarm came first. Reading left to "
                    "right is walking from this notebook's protocol to the one the "
                    "nearest published work uses, and the probe is identical at "
                    "every point._\n\n"
                    "_What the two panels do **not** show is the price, which is in "
                    "the precision column of the table above. The rung that buys "
                    "most of the movement also spends thirty times the false-alarm "
                    "budget, and the rung after it hides that by rebalancing the "
                    "population the precision is measured on._"
                ),
            ]
        )

    bridge_ladder_figure()
    return


@app.cell
def _(mo):
    mo.md("""
    ## How large is the pooling term on its own?

    Span width is the one convention with a free parameter, so it is worth
    sweeping rather than asserting. Pooling buys coverage because a span of
    weak-but-consistent evidence clears a threshold that no single token in it
    would, and it costs lead time because the span's verdict is not available
    until the span has finished.
    """)
    return


@app.cell
def _(
    BRIDGE,
    FIGURES,
    INK_MUTED,
    SERIES,
    bridge_pick,
    missing,
    mo,
    pd,
    plt,
    save,
    tidy,
):
    def bridge_span_figure():
        frame = pd.read_parquet(bridge_pick.value)
        rows = []
        for span in (8, 16, 32, 64, 128):
            table = BRIDGE.build(frame, span, [0.01, 0.30], draws=1, seed=42)
            pooled = table[table["step"] == "span pooling only"]
            base = table[table["step"] == "0. this protocol"]
            if pooled.empty or base.empty:
                continue
            rows.append(
                {
                    "span": span,
                    "warning": float(pooled["warning_256"].iloc[0]),
                    "baseline": float(base["warning_256"].iloc[0]),
                    "offset": float(pooled["median_offset"].iloc[0]),
                    "offset_baseline": float(base["median_offset"].iloc[0]),
                }
            )
        if not rows:
            return missing("Nothing to sweep: the ladder produced no pooled rung.")
        swept = pd.DataFrame(rows)

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
        axes[0].plot(swept["span"], swept["warning"], "o-", color=SERIES[0], linewidth=1.8)
        axes[0].axhline(
            swept["baseline"].iloc[0], color=INK_MUTED, linestyle="--", linewidth=1
        )
        axes[0].set_ylabel("coverage of the run-up")
        axes[0].set_ylim(bottom=0)

        axes[1].plot(swept["span"], swept["offset"], "o-", color=SERIES[1], linewidth=1.8)
        axes[1].axhline(
            swept["offset_baseline"].iloc[0], color=INK_MUTED, linestyle="--", linewidth=1
        )
        axes[1].set_ylabel("median first alarm, tokens from the loop")

        for axis in axes:
            axis.set_xscale("log", base=2)
            axis.set_xticks(swept["span"])
            axis.set_xticklabels([str(int(s)) for s in swept["span"]])
            axis.set_xlabel("tokens per pooled span")
            tidy(axis, xgrid=False)

        fig.suptitle(
            "What pooling buys, and what it costs", x=0.005, ha="left",
            fontsize=10, color=INK_MUTED,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return mo.vstack(
            [
                save(fig, "protocol_bridge_span", FIGURES),
                mo.md(
                    "_Both panels are at the same 1% budget as the rest of this "
                    "notebook, so only the aggregation changes. The dashed line is "
                    "the per-token decision, the leftmost rung of the ladder._\n\n"
                    "_Coverage rises with the span and the alarm moves later, which "
                    "is the trade the span width controls. Neither curve is a "
                    "property of the probe; both are properties of how its output "
                    "is read._"
                ),
            ]
        )

    bridge_span_figure()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The accumulation rung, written out

    The third rung is the cumulative-sum change detector of Page (1954), the
    same rule the nearest published work accumulates its classifier scores
    with. For the mean probe score $x_i$ over span $i$,

    $$S_i = \max\left(0,\ S_{i-1} + (x_i - k)\right), \qquad S_0 = 0$$

    Evidence above the reference level $k$ accumulates and evidence below it
    decays, and the floor at zero stops a quiet stretch from building up credit
    that a later rise would have to spend before firing. So an isolated spike
    dies immediately while a sustained rise compounds, which is the whole point
    of accumulating rather than thresholding.

    Two choices here are ours rather than the method's, and both are worth
    stating. The reference level $k$ is measured rather than tuned: it is the
    median pooled score over healthy answers, so *no evidence* means *behaving
    like a typical healthy span*. And the firing threshold is solved for the
    false-alarm budget, exactly as it is for every other row, rather than set
    from average-run-length arguments as a textbook cumulative sum would. That
    second choice is what keeps the rungs comparable: every row spends the same
    share of healthy answers.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## What this settles, and what it does not

    **Most of the distance between published early-warning numbers is
    convention.** The same probe, unchanged, moves from flagging a small
    fraction of the run-up and firing after the loop has started, to flagging
    most of it and firing hundreds of tokens ahead. Nothing was learned in
    between. Any comparison of lead times across papers that does not fix the
    operating point, the aggregation and the base rate is comparing reporting
    choices.

    **The operating point is the largest single term**, which is also the one
    least often stated plainly. Pooling and accumulation matter, and the base
    rate matters mainly for precision, but the budget dominates all three.

    **The cost is in the precision column, and rebalancing hides it.** At the
    loose operating point most alarms on this population are false. The same
    detector reads far better once the healthy answers are thinned to match the
    degenerate ones, which is what a balanced evaluation set does. Neither
    number is wrong; they describe different populations, and only one of them
    is the population a deployment sees.

    **A rebalanced set cannot resolve a strict budget at all.** Thinned to the
    number of degenerate answers, one percent of the healthy ones is a single
    answer, so the threshold is set by the largest of a small sample and lands
    lower than the same nominal budget on the full population.

    **What this is not.** It is not an evaluation of anyone else's detector, and
    it cannot be read as one. Their model may well do better than ours on their
    own data. Two further limits are worth carrying: a span here is a fixed
    number of tokens rather than a sentence, because the stored scores carry no
    text, and the reference level of the cumulative sum is a choice. So this is
    *a* path between two protocols rather than *the* path, and the span sweep
    above is the evidence that its shape is not an artefact of one setting.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Part 6. What the nearest prior work actually measures

    Every figure this project attributes to another paper was checked against
    that paper's own text, not against a summary of it. The reason to record it
    here is that two of the claims are load-bearing: the paper argues that the
    metric this area reports is saturated, and that the works nearest to it read
    the final layer. Both needed checking before anything was built on them.

    `docs/paper/references.bib` carries the verified figures as comments on each
    entry, so none of this has to be re-derived from a secondary reading.

    ## What the four nearest detectors measure, and where

    | | unit measured | representation | decoding | needs the token cap |
    |---|---|---|---|---|
    | Duan et al. 2026 | **sentence** (statement loops), token (numerical) | mean of last-layer states | greedy | no |
    | Xie et al. 2025 | **chunk**, at the double-newline boundary | output of the final block | temperature 0 | no |
    | Yu et al. 2025 | **whole response** | MLP activation similarity, **every** decoder layer | temperature 0 | **yes**, `max_new_tokens` |
    | LoopGuard 2026 | per decoding step, window 256 | none, it is a text statistic | not stated | **yes**, length $\geq$ 2480 of 2500 |

    Two consequences follow, and both changed how the paper is written.

    **"Answer-level" is wrong as a description of this literature.** Only Yu et
    al. classify a whole response. Duan et al. work per sentence and per token,
    Xie et al. per chunk. The saturation claim is real but it is about accuracy
    on *units drawn from inside the loop*, whatever the unit is, because in every
    case the unit population is dominated by tokens already deep in the pattern.
    That is the phrasing to use.

    **The final-layer claim holds only for the probes.** Duan et al. take the
    mean of last-layer hidden states, and Xie et al. read the output of the final
    transformer block. Both confirmed. Yu et al. do not: they build features from
    activation similarities across every decoder layer and feed them to a
    three-layer network, so they are a different kind of method rather than a
    counterexample. The depth argument is scoped to work that trains a probe on a
    hidden state.

    Duan et al. do run a layer-wise analysis, but it is descriptive, reporting
    cosine similarity and L2 distance across depth. Probe quality is never
    reported as a function of depth, which is what the depth result here is about.

    ## Is the reported metric actually saturated? Measured at their units

    Measured rather than asserted, at each of the three granularities, on
    validation, from the same stored per-token scores everything else uses. A unit
    is positive when it begins at or after the frontier; units from healthy
    answers are all negative. AUC is on the natural population. Accuracy is the
    best achievable on an equal-sized subsample of the two classes, which is the
    most generous reading of what a balanced test set reports.

    The units match the granularities of the nearest prior work. The *operators*
    do not reimplement anyone's detector, and
    `degeneration_probe/analysis/unit_levels.py` states where they differ: the
    chunk reading is faithful, the sentence reading is a proxy, and the answer
    reading is this project's own maximum over tokens.

    ```bash
    python scripts/unit_level_saturation.py --split val   --build-root $BUILD --tokenizer $TOKENIZER_JSON   --scores "recipe/S1 #1 rollout_balanced W=128 @L15=outputs/<run>/latest/layers/layer_15/scores/val.parquet" ...
    ```

    Names split at the last `=`, because a recipe label contains one. Written to
    `outputs/analysis/unit_level_saturation_val.csv`, which is what
    `notebooks/paper.py` reads.

    **Twelve leading recipes, each at the depth it was selected at.** Selection
    strategy, window width, label family, horizon and class balance all vary
    across these rows.

    | recipe | sent. AUC | sent. acc | chunk AUC | chunk acc | answer AUC | answer acc |
    |---|---|---|---|---|---|---|
    | S1 #1 rollout_balanced W=128 @L15 | 0.9962 | 0.9711 | 0.9956 | 0.9679 | 0.9998 | **1.0000** |
    | S1 #2 frontier_window W=256 @L15 | 0.9943 | 0.9688 | 0.9946 | 0.9645 | 0.9996 | 0.9907 |
    | S2b #1 soft exp/256 @L15 | 0.9936 | 0.9676 | 0.9942 | 0.9628 | 0.9996 | 0.9907 |
    | S2a #2 all_tokens h=1024 @L15 | 0.9932 | 0.9689 | 0.9938 | 0.9629 | 0.9997 | 0.9954 |
    | S2b #2 soft linear/256 @L15 | 0.9930 | 0.9668 | 0.9937 | 0.9607 | 0.9995 | 0.9907 |
    | S2b #3 soft exp/128 @L15 | 0.9928 | 0.9668 | 0.9935 | 0.9597 | 0.9995 | 0.9907 |
    | S2a #1 frontier W=512 h=256 @L15 | 0.9925 | 0.9657 | 0.9933 | 0.9592 | 0.9994 | 0.9907 |
    | S2a #3 frontier W=512 trail h=512 @L15 | 0.9909 | 0.9633 | 0.9916 | 0.9554 | 0.9993 | 0.9907 |
    | S1 #3 hard_negative W=128 @L4 | 0.9892 | 0.9580 | 0.9834 | 0.9429 | 0.9991 | 0.9954 |
    | S2d #2 pos_weight off @L4 | 0.9892 | 0.9585 | 0.9833 | 0.9389 | 0.9988 | **1.0000** |
    | S2d #3 positive_fraction 0.5 @L4 | 0.9886 | 0.9580 | 0.9806 | 0.9319 | 0.9984 | **1.0000** |
    | S2d #1 pos_weight on @L4 | 0.9878 | 0.9556 | 0.9802 | 0.9337 | 0.9984 | 0.9907 |

    Range across all twelve: **0.0084** sentence AUC, **0.0154** chunk,
    **0.0014** answer. At answer level three of the twelve reach accuracy exactly
    1.0000. Warning coverage separates these same twelve recipes clearly, which is
    what the stages above are about, so this is the claim in its strongest form:
    the quantity cannot rank the recipes at all.

    **One recipe, read at every depth it was scored at.**

    | depth | sent. AUC | sent. acc | chunk AUC | chunk acc | answer AUC | answer acc |
    |---|---|---|---|---|---|---|
    | layer 12 | 0.9924 | 0.9648 | 0.9930 | 0.9616 | 0.9991 | 0.9861 |
    | layer 8 | 0.9890 | 0.9589 | 0.9898 | 0.9573 | 0.9984 | 0.9954 |
    | layer 20 | 0.9886 | 0.9592 | 0.9888 | 0.9518 | 0.9984 | 0.9907 |
    | layer 4 | 0.9878 | 0.9556 | 0.9802 | 0.9337 | 0.9984 | 0.9907 |
    | layer 30 | 0.9841 | 0.9501 | 0.9713 | 0.9300 | 0.9876 | 0.9722 |

    Range 0.0083 sentence AUC, 0.0217 chunk, 0.0115 answer, over depths whose
    coverage of the run-up differs by an order of magnitude. Most of even that
    comes from layer 30; layers 4 to 20 span 0.0046 at sentence level.

    **Model-free scorers, for scale.**

    | scorer | sent. AUC | chunk AUC | answer AUC |
    |---|---|---|---|
    | entropy, trailing 128 | 0.9150 | 0.9042 | **0.9915** |
    | repetition, trailing | 0.8983 | 0.8821 | 0.8943 |
    | repetition, forward | 0.8775 | 0.8699 | 0.9027 |
    | rep_l, l=128 | 0.8695 | 0.8223 | 0.8676 |
    | lrs | 0.6968 | 0.6645 | 0.5947 |

    **What this settles.** The quantity is not uninformative: it separates a probe
    from a text statistic by twenty to thirty points, and the model-free rows
    spread over 0.22 to 0.40 AUC where the probe rows spread over 0.001 to 0.02.
    It is saturated across exactly the choices that decide early warning while
    remaining sensitive to whether a scorer works at all. So it belongs in the
    protocol as a floor to clear, and it cannot be used as a ranking or to select
    a depth.

    **The numbers land where the published ones do**, which is the evidence for
    taking those results as settled and reproducing them rather than disputing
    them. Xie et al. report 89.8 to 93.5% accuracy at chunk level and these
    recipes reach 93.2 to 96.8%. Duan et al. report 0.998 to 1.000 AUC at sentence
    level and these reach 0.988 to 0.996. Yu et al. report 95.24% accuracy per
    response and these reach 99.1 to 100%.

    **The answer unit is the one length contaminates.** Windowed entropy reads
    0.9915 AUC per answer and 0.90 to 0.92 per sentence or chunk. It is the only
    scorer whose ranking improves markedly as the unit grows, which is what a
    length artefact looks like: a maximum over tokens accumulates with length, and
    only the answer unit is free to grow. The shorter units do not have that
    freedom, which is a second reason to prefer them.

    Two caveats on the measurement. The probe rows cover 3,634 of the 3,640
    validation answers, because that is what the probe score files hold. And these
    probe numbers are **unattributed**: the score directories carry no
    `scoring_provenance.json`, so the checkpoint behind them cannot be named.
    `scripts/audit_score_provenance.py` reports 371 of 377 scored directories in
    that state, five of which are model-free baselines with no checkpoint to
    record.

    ## Which answer-level metric is saturated, and which is not

    The saturation claim is about the quantity other work reports, so it has to be
    made with that quantity in hand rather than asserted about it. Measured on our
    own probes, at answer level, on validation: 108 degenerate answers of 3,634.

    Start with the floor. The positive rate is 2.97%, so **a scorer that fires on
    nothing already scores 0.9703 accuracy**. That is the number every accuracy below
    has to be read against, and it is most of the way to the best probe.

    **The twelve leading recipes, each at its own selected depth:**

    | | AUC | AP | accuracy | balanced acc |
    |---|---|---|---|---|
    | best | 0.9998 | 0.9929 | 0.9986 | 1.0000 |
    | worst | 0.9984 | 0.9552 | 0.9934 | 0.9907 |
    | **range** | **0.0014** | **0.0377** | **0.0052** | 0.0093 |

    One recipe across five depths: AUC range 0.0115, AP range **0.1662**, accuracy
    range 0.0107.

    **Two of the three are saturated, and one is not.** Accuracy is nearly inert: the
    whole distance between firing on nothing and the best probe is 2.8 points, and the
    longest-repeated-substring scorer, which never fires at any budget in use here,
    scores 0.9703 and therefore lands within 2.8 points of the best probe. AUC moves
    by 0.0014 across twelve recipes that differ in selection rule, window width, label
    family, horizon and class balance.

    Average precision is the exception, and it is worth understanding why. With
    positives at 3% of the split, AUC is dominated by the enormous mass of easy
    negatives, while average precision reads the top of the ranking, which is the only
    part anything is decided on. It spans 27 times the AUC range across recipes and 14
    times it across depths.

    **Does the one with dynamic range actually rank correctly?** Against warning
    coverage 256, the rule's own objective:

    | metric | Spearman, recipes | picks | cost | Spearman, depths | cost |
    |---|---|---|---|---|---|
    | AUC | 0.580 | S1 #1 | 1.50x | 0.400 | 1.16x |
    | accuracy | 0.607 | S1 #1 | 1.50x | 0.700 | 1.16x |
    | balanced accuracy | **-0.509** | S1 #1 | 1.50x | 0.205 | 1.39x |
    | average precision | 0.657 | **S2a #2** | **1.00x** | 0.400 | 1.16x |

    Average precision picks the best recipe. It does not repeat that on the depth
    axis, where it chooses layer 12 against warning coverage's layer 4, and its rank
    correlation is 0.657 and 0.400, which is not the agreement of a substitute. The
    recipe hit is best read as inside the noise: the top two recipes are separated by
    0.0001 in warning coverage.

    **Balanced accuracy is worse than uninformative, it is inverted.** Its rank
    correlation with warning coverage is negative, because three of the recipes with
    the *lowest* coverage of the run-up reach exactly 1.0000. That matters beyond this
    project: Duan et al. and Xie et al. both report accuracy on balanced test sets.

    So the claim to make is narrower and stronger than "answer-level metrics are
    saturated". The metrics this literature reports, accuracy and AUC, are saturated
    and mis-rank, and accuracy on a balanced set anti-correlates with what is wanted.
    Average precision keeps its dynamic range and nobody reports it.

    **A bug found while adding these.** The sweep behind every threshold-based metric
    here allowed a cut *inside* a group of equal scores, which no threshold can do,
    and so flattered accuracy and average precision wherever ties were common. All
    four metrics now run off one tie-aware sweep over distinct score values, checked
    against scikit-learn and against brute force with heavy ties. The probe figures
    did not move, since a float32 score is effectively continuous; a step-function
    scorer like the longest repeated substring is where it would have shown.

    ## Figures confirmed

    - Duan et al.: early detection rate 0.64 to 0.76, false positive rate 0.24 to
      0.34, 36.5 to 51.4 sentences and **1305.9 to 1870.2 tokens** of lead.
      Linear, SVM and MLP classifiers reach 0.991 to 0.999 accuracy and 0.998 to
      1.000 AUC. Greedy decoding. Balanced test sets of at least 50 loop and 50
      non-loop cases per model, with 50 non-loop instances held for calibration.
      Loop labels are $k \cdot l > 500$ tokens, or more than three sentence
      repetitions.
    - Xie et al.: at temperature 0 on Qwen-7B, **89.77 to 93.52% accuracy and
      95.84 to 98.63 AUROC** across four benchmarks. At temperature 0.6 accuracy
      falls as low as 77.96%. Chunks are labelled by embedding similarity at 0.99
      using all-MiniLM-L6-v2. Their appendix tracks one problem over 221 chunks
      with the classifier score rising from 1.19e-10 to 1.000, which is the
      evidence that detection strength grows *inside* a loop rather than before it.
    - Yu et al.: 95.24% accuracy, 0.87 F1, 2.59% false positive rate, 82.88%
      recall, averaged over six models. A **400-token warm-up** is required before
      the detector can fire, chosen as the 78th percentile of response length.
      Roughly 680 test responses.
    - LoopGuard: type-token ratio at most 0.2, compression ratio at most 0.12,
      length at least 2480. Sliding window of **256 tokens**. Online trigger is a
      debounced three-way vote plus a top-1 probability streak, 0.9 for six steps.
    - Kramár et al.: the short-to-long-context shift is the paper's stated
      problem, and it proposes architectures for it.

    ## Figures that did not hold

    - Duan et al.'s token lead time tops out at **1870**, not 2000.
    - Xie et al.'s AUROC tops out at **98.63**, not 99, and the accuracy range
      comes from one model across four benchmarks rather than from several models.
    - `rep_l` is not "the share of tokens already occurring in the previous $l$
      tokens". Welleck et al. define it on the model's **argmax next-token
      prediction**. At temperature 0.7 the two differ, so the baseline here is a
      variant and says so where it is introduced.

    ## Metadata that was wrong

    Four entries had invented authors or titles, which is worth knowing as a
    pattern rather than as four separate slips.

    - LoopGuard's given names were wrong throughout. The authors are Dongjie Xu,
      Hao Wu, Weijie Shi, Yue Cui and others.
    - The Look-back paper had both a wrong title and wrong authors. It is
      "Look-back Decoding for Open-Ended Text Generation" by Xu, Zhou,
      Celikyilmaz and Ma, EMNLP 2023, pages 1039 to 1050.
    - "Learning to Break the Loop" had four of six authors wrong.
    - Tian Lan appeared as "Yixuan Lan"; Bin Gu appeared as "Qiaoyu Gu".

    ## What could not be checked

    SpecRA is behind a bot check on OpenReview. Its authors, venue and acceptance
    status are unverified and none of its mechanism should be quoted. Its public
    summary claims *theoretical* false-positive bounds, so the claim that no
    deployed rule reports a false-alarm rate against an independent reference has
    to be narrowed to empirical rates.
    """)
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
      selection_history.parquet         the rule's record at every evaluation
      selection_outcomes.json           where each depth stopped, and its pick
      checkpoint_replay.parquet         the same record, recovered afterwards
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

    **Ask for the walltime a job needs, not a round number.** The queue is the
    binding constraint, and a short job is backfilled into a gap that a long one
    waits behind, so padding a limit costs hours of waiting rather than buying
    safety. Measured from completed jobs, a scoring pass is about **eight minutes
    of container and model startup plus three minutes per layer or checkpoint**:
    three layers land in 14 to 18 minutes, seven in 25. So eight pairs fit inside
    40 minutes and forty do not fit inside three hours.

    Split a long list rather than raising the limit. Forty checkpoints as four
    jobs of ten, each asking 45 minutes, start far sooner than one job asking
    three hours, and the first results arrive while the rest are still queued.
    `sacct -u $USER --format=JobID,JobName%34,Elapsed,State,Timelimit` is where the
    numbers above come from, and is the thing to re-check before assuming them.

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

    **Scores are stored at single precision, and that is load-bearing.** Half
    precision has a spacing of about $5 	imes 10^{-4}$ just below one, so any
    score within that distance of the top of the range collapses onto it. Ties at
    a scorer's ceiling are what make a small false-alarm budget unspendable, so
    half precision manufactures the failure the protocol reports. The cast lives
    in three places and all three matter: `build_scores`, and the record built by
    each of `score_baselines.py` and `score_rollouts.py`.

    **The labelling pipeline's repetition score reads the window *ahead* of the
    token it labels**, over $[t,\ t+256)$. That is correct for annotating a
    finished answer and impossible for a live monitor. Use
    `repetition_trailing` for any comparison against a probe; `repetition` stays
    on the tables as the reference row that shows what the lookahead is worth.
    Nobody else made this mistake, so it is not a criticism of the literature.

    **`llama_nemotron` is code.** It is the `code` split of the Llama-Nemotron
    SFT set, which the domain name does not say. So the in-domain corpus contains
    a code source and `codeforces`, held out, is a second one. The held-out
    result is a test of generalisation across sources *within code*, not across
    kinds of text.

    **`test_indomain` has never been scored for any probe.** The baselines have
    it, because a model-free scorer fits nothing and so cannot leak. Do not touch
    it for a probe without deciding to spend it.

    **Known traps.**

    - A trailing window with horizon 0 contains no positive token. The dataset
      refuses to build it rather than training on nothing.
    - `val/loss` carries a class weight fitted to each recipe's own training
      stream, so it is not comparable between recipes. Use
      `val/loss_unweighted`.
    - The rule's record is only as firm as the healthy answers behind its
      threshold. At a one percent false-alarm budget, four hundred answers put
      the threshold on the top three of them and the objective derived from it
      carries a relative spread of 46%; the whole split puts it on the top
      thirty-five. A run judged from its own evaluations has to monitor the
      whole split, and the trainer says at startup how many answers its
      threshold is resting on.
    - Rollout-level recall is not a selection metric on this data. With a
      hundred degenerate answers it reads the same at every step of a run, so
      the checkpoint it names is whichever one caught one more answer.
    - The token budget per step is derived from the *measured* tokens per
      example, not from the configured window size. Wide windows are clipped by
      the ends of the answers they sit in, so they land furthest below the
      request. The run table reports what each step really saw.
    - Most runs were interrupted by the walltime and resumed, so a run's
      recorded duration covers only its final leg, not its real cost.
    - `run_info.json` does not record the positive fraction or the decay length,
      so those are read back out of `resolved_config.json`.

    **A capped answer the judge could not rule on belongs to neither class, and
    two code paths used to disagree about that.** The onset table marks 72 capped
    answers with no resolved onset: 63 `judge_failed`, 1 `not_found`, and 8
    `not_degenerating`. Only the last 8 were actually ruled on, and they are
    legitimately negative. The other 64 are unknown.

    The pinned evaluation population already left the unknown ones out, so a probe
    reads 3,634 validation answers, 108 degenerate and 3,526 negative, and every
    probe number and every replayed checkpoint has always been measured on that.
    The model-free baselines did not: they build their population from the onset
    label table directly, which put 6 unjudged answers from `if_sft_data_verified`
    among the healthy ones on validation.

    Those 6 are not borderline. All reached the 4096-token cap, and their peak
    trailing repetition score has a median of 0.804, above the median of the
    confirmed positives (0.741) and above the 99th percentile of answers that ended
    at an end-of-sequence token (0.756). At a 1% budget the trailing repetition
    score raised 35 false alarms and 4 of them were these answers; for windowed
    entropy it was 5 of 35. So more than a tenth of the tightest operating point
    was being spent on answers that are probably positives, which lifted the
    threshold and suppressed everything measured above it.

    Excluding them moved the baselines and nothing else:

    | at a 1% budget | trailing repetition | windowed entropy |
    |---|---|---|
    | recall | 0.4815 to 0.4907 | 0.5370 to **0.6852** |
    | in-pattern | 0.4303 to 0.4328 | 0.1582 to 0.2283 |
    | warning 256 | 0.0291 to 0.0313 | 0.0039 to 0.0055 |
    | never fired | 56 to 55 | 50 to **34** |

    Small for a scorer whose healthy scores are spread out, large for one whose
    scores bunch near the ceiling.

    The exclusion now lives in one place, `read_scores`, keyed on the resolution
    rather than on whether an answer reached the cap, so the two paths cannot
    diverge again. `evaluate_scores.py` finds the label table from the run's own
    configuration and refuses to run when it cannot, rather than quietly keeping
    the old population; `--keep-unjudged` reproduces a number measured before the
    change.

    **Scoring inherits a filter that belongs to training, and it is not
    exhaustive.** `score_rollouts.py` builds its rollout list from the training
    dataset, which drops any rollout the label family cannot label: first when
    `derive_targets` returns nothing, then when the resulting target array holds no
    finite value. Both are right for training, since there is nothing to learn from
    such a rollout. Neither is right for scoring, where a probe can score any token
    and the false-alarm rate is a rate over the whole healthy population.

    For the frontier families the two effects coincide harmlessly: the dropped
    rollouts are the capped ones with no resolved onset, which is why a probe table
    holds 3,634 of 3,640 validation answers and why those tables were already on the
    right population. For the `token_signal` family it does real damage. The
    labelling pipeline's repetition score is NaN wherever a full 256-token window
    does not fit, so an answer shorter than the window has no score anywhere, and
    **846 validation answers, all of them negative, are silently dropped**. That is a
    quarter of the healthy population and it is biased toward short answers, which
    are the easiest negatives. Every threshold solved on such a table is solved over
    a population missing its easy cases.

    Nine score directories are affected, all of them the regression recipe under
    LoRA-all across three seeds, so nothing reported here reads from them. The
    in-flight LoRA reruns include that arm and will reproduce it unless scoring stops
    reusing the training dataset's filter.

    `evaluate_scores.py` now compares a scored table against the split it claims to
    cover, prints how many rollouts are missing, and records the shortfall in
    `decision_thresholds.json` so the fact travels with the numbers instead of being
    invisible. That guarantee was stated in a docstring and never checked.

        **A cross-corpus directory must be told which corpus it describes.** Scores
    under `cross_model/` or `lora_transplant/` were computed against another model's
    build, while the `resolved_config.json` above them names the build the run was
    *trained* on. Which rollouts a judge could not rule on differs between corpora,
    so reading the nearer configuration drops the wrong rollouts: applying the
    Apertus exclusions to the Apertus 1.5 tables removed 5 and 6 legitimate
    rollouts, one of them a positive, which moves a recall denominator.

    Those tables already leave their own unjudged rollouts out at scoring time, so
    the correct additional exclusion for them is empty: 3,621 of 3,640 scored for
    one target and 3,595 of 3,640 for the other. `evaluate_scores.py` now refuses to
    guess a label table for any path under those two markers and asks for
    `--onset-labels`, rather than silently reading the one next door.

    The 72 directories under `cross_model/_superseded_pilot1024/` have no label table
    at all. They are evaluated with `--keep-unjudged` so they stay internally
    consistent, and they are on a superseded build, so nothing in the paper should
    read from them.

        **Count a population by resolution, not by stop reason.** On validation the
    number of answers that ended at an end-of-sequence token is 3,526, which is
    exactly the pinned negative count, so the two agree by coincidence. On
    `test_indomain` they do not: 3,514 against the correct 3,520, because that
    split holds 6 answers the judge ruled healthy and 13 it could not rule on, and
    only the second group should go.

    **Where else to look.** `notebooks/inspect_runs.py` is the inventory of
    every run regardless of experiment. `notebooks/rollout_metric_explorer.py`
    plots a single answer's metrics against its frontier, which is the fastest
    way to sanity-check a label by eye. `notebooks/paper.py` builds every table
    and figure the paper prints, so no number in it is typed by hand.
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
