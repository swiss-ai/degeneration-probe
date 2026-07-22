# Plan: unified evaluation methodology for Entropy / TTR / LRS vs. LLM judge

Status: agreed in discussion, not yet implemented. Written down before a
`/compact` so the design survives context loss. Scope spans both repos:
`degeneration-probe` (notebooks) and `degeneration-probe-doc` (`main.tex`).

## Why (current state, as of this session)

The three structural metrics currently use inconsistent populations/protocols:

- **LRS** (`inspect_dataset.ipynb` cell 30, table `tab:lrs-confusion`): population
  is `calibration_sample.parquet` (== *all* rollouts with `stop_reason != "eos"`,
  not a subsample) filtered to `stratum == "truncated"`, joined with successful
  judge verdicts. Ground truth = judge `is_degenerating`. Prediction =
  `onset_period_repeat_count >= MIN_PERIOD_REPEAT_COUNT` (3, fixed a priori).
  No calibration/test split (nothing is tuned). Aggregate row is a simple pool
  across domains ("totaled across domains").
- **TTR** (`ttr_inspection.ipynb`): same truncated-only population and pooling
  behavior as LRS. Threshold 0.8 also fixed a priori.
- **Entropy** (`entropy_inspection.ipynb`): population combines currently-judged
  truncated rollouts with *historically judged natural-EOS rollouts* — leftover
  judge verdicts in `results_*.parquet` from a now-removed `flagged_natural_eos`
  stratum (dropped from `select_calibration_sample` in commit `bd4ed0c`,
  2026-07-21, because `stop_reason == "length"` alone was already a strong
  proxy and judging flagged-EOS rows too wasn't worth the budget). Needs a
  calibration/test split by prompt hash because its threshold *is* tuned
  (balanced-accuracy maximization) — this is the one part of the current setup
  that's methodologically necessary, unlike LRS/TTR's fixed thresholds.

This produces three different tables with different columns (LRS/TTR:
Precision/Recall/Accuracy; Entropy: Precision/Recall/Specificity/Bal.Acc/MCC),
different populations, and no per-domain vs. aggregate distinction beyond a
straight pool.

## New unified population (agreed)

- **Split-agnostic**: none of this needs to respect the train/val/test split
  used elsewhere for probe training — that's an orthogonal concern. The only
  place a split still matters is Entropy's own threshold *tuning* step (see
  below), because that's the only place something is fit to data.
- **Positives** = every rollout (any domain, any split) with a successful
  LLM-judge verdict `is_degenerating == True` **and** a locatable
  `onset_quote` (`find_string_in_tokens` succeeds). This drops only ~0.4% of
  true positives (match rate is 99.6-100%, per the Reliability section) but
  means the binary-detection population and the onset-localization population
  become the same set, instead of two separately constructed ones.
- **Negatives** = every rollout with `stop_reason == "eos"`, treated as
  negative *by definition*, no judge call needed. Justification already in the
  doc: LRS's own match-rate analysis (`tab:lrs-falsepos`) shows EOS rollouts
  have only short incidental repeats (median 20-24 tokens) vs. 1475-1716
  tokens under the cap — sustained loops essentially never let a model
  naturally stop, so EOS-as-negative is a safe proxy, not just a convenient
  one.
  - Also **keep** the small number of judge-confirmed-negative rows among the
    truncated population (the ~0.5-7% mentioned in "Cost and feasibility") as
    extra hard negatives — already paid for, more informative than bulk EOS
    negatives, don't discard them.
- **No dependence on any historical/legacy judge data.** Once negatives come
  purely from `stop_reason == "eos"`, the whole `flagged_natural_eos` /
  "historically judged EOS" story becomes moot for this analysis — those old
  verdicts are no longer needed at all. This resolves the earlier confusion
  about that stratum without requiring any doc explanation of its history.
- **Revised, per user feedback (2026-07-22): all three metrics get a
  calibration/test split, not just Entropy.** LRS's `MIN_PERIOD_REPEAT_COUNT
  = 3` and TTR's `1-TTR > 0.8` are currently fixed a priori, with no data
  used to justify the specific number — that's methodologically weaker than
  Entropy's tuned threshold, not stronger, so "LRS/TTR need no split" was
  backwards. Fix: reuse the exact same deterministic prompt-hash split
  Entropy already uses
  (`hashlib.sha256(f"entropy-inspection-v1:{prompt_id}")`, generalize the
  literal tag) for all three metrics. For each metric: pick its threshold
  (LRS: sweep the integer repeat-count cutoff; TTR: sweep the score cutoff;
  Entropy: sweep `tau` as today) by maximizing balanced accuracy on the
  calibration half, then report every number in the paper (confusion
  metrics, per-domain tables, onset offsets) on the held-out test half only.
  This also means LRS/TTR's *reported* numbers change from "fit on
  everything" to "fit on calibration, scored on test" — expect a small
  shift, and it's the more defensible number. One split, reused for all
  three metrics and for the onset-offset comparison — not three separate
  splits.

## Domain balancing (agreed)

- **Per-domain rows**: always show `n` (or TP/FP/FN/TN) next to every metric
  so small/unreliable domains (e.g. `medical_o1`, which has very few positive
  rollouts) are visibly flagged rather than hidden. Consider annotating rows
  below some minimum n (open question: threshold value, e.g. `n_positive < 10`)
  as "insufficient data" instead of computing a possibly-NaN or single-sample
  metric.
- **Aggregate "ALL domains" row**: switch from simple pooling (today: weighted
  by however many rollouts each domain happened to produce) to a
  domain-balanced aggregate, e.g. a macro-average of per-domain metrics (equal
  weight per domain) rather than physically subsampling rows. Rationale:
  today's pooled aggregate is dominated by whichever domain has the most
  rollouts, which is an artifact of data collection volume, not of how well a
  rule generalizes across problem types. Macro-averaging is deterministic and
  avoids adding sampling variance for no benefit.

## Views to test per metric (2026-07-22 addition)

`docs/analysis/probe_eval_and_training_protocol.md` §1 defines three
complementary "views" for evaluating a trained probe against the
degeneration frontier. The same three views are the right lens for
Entropy/TTR/LRS-as-detectors too, so the plan should explicitly cover all
three, not just rollout-level detection. **Never cite that document (or the
term "frontier"/"view A/B/C") from `main.tex`** — restate each view in
self-contained terms there, since the paper's readers don't have that
context; this section is for our own tracking only.

- **View A — rollout-level detection.** Already covered by the plan above:
  confusion-matrix / precision / recall / specificity / balanced accuracy /
  MCC / accuracy, aggregate + per-domain. Addition: also report a
  threshold-free ROC/PR curve (sweep the metric's own knob — repeat-count
  cutoff for LRS, score cutoff for TTR, tau for Entropy) and its AUC, as a
  sanity check that the calibrated operating point wasn't a lucky pick —
  supplements the single fixed operating point, doesn't replace it.
- **View B — token-level coverage. Currently a gap, not computed for any of
  the three metrics.** Needed, per metric:
  - On negative (EOS) rollouts: the *token-level* false-positive rate
    (fraction of individual tokens flagged, not just "did any token in the
    rollout cross threshold"), plus the fraction of rollouts with at least
    one such false positive.
  - On positive rollouts, restricted to tokens at/after the judge-located
    onset (`onset_quote` position): *recall* — fraction of in-pattern
    tokens the metric also flags — as a rate, plus, per rollout, a count of
    missed in-pattern tokens (missing one token deep in a long loop is a
    different failure than missing its last 50).
  - Tokens strictly before onset are not scored against a positive label
    (not degenerate yet); they belong to View C instead.
  - This is the most new plumbing for **LRS** specifically, since today it
    only produces a rollout-level match count, not a per-token flag —
    TTR/Entropy already have a per-token score to threshold.
- **View C — early-stopping lead time.** Already planned via the shared
  onset-comparison table: signed/absolute distance between each metric's
  first-alarm position and judge onset (median/IQR) for true positives,
  reported separately for true positives where the metric never fires. New
  relative to what's in the notebooks today: also report, for true
  negatives, the rate at which the metric fires at all (false early-stop
  rate) — currently the onset-offset analysis only looks at true positives.

Net: View A and View C are already in scope; View B is a genuine gap worth
an explicit decision (see open questions) on whether to include it in this
pass or defer it, given the extra LRS plumbing it needs.

## Structural changes to `main.tex` (agreed direction)

Move from "one Empirical Performance subsubsection per metric" (current:
Entropy §2.1.3, TTR §2.2.7, LRS §2.3.5) to:

1. All metric **definitions/algorithms only** stay where they are (Entropy
   §2.1 Definition+Algorithm, TTR §2.2 Definition+...+Algorithm, LRS §2.3
   Motivation+Digit normalization+Algorithm+Why not tolerate). Remove each
   metric's own "Empirical performance" subsubsection.
2. LLM Judge section (§2.4) stays as-is (Setup, System prompt and output, Why
   we trust it, Cost and feasibility, Reliability) — it's the reference
   method and must be explained before results that depend on it.
3. **One new unified "Empirical Performance" section**, after LLM Judge,
   containing:
   - One paragraph stating the shared population/protocol described above
     (written once, not three times).
   - One master aggregate table: rows = metric (Entropy/TTR/LRS) × dataset
     build, same column set for all three now that all three have real
     negatives (Precision/Recall/Specificity/Bal.Acc/MCC/Accuracy).
   - One master per-domain table (or one per dataset build) with the same
     metrics broken down by domain, `n` shown, and domains under the min-n
     line (§ open questions #1) marked rather than given a computed number
     (`medical_o1`: 0 test positives — "no positive rollouts in test";
     `aime_2025`/`codeforces`: below-threshold `n`, reported but flagged).
   - One ROC or PR curve figure (whichever is more informative given the
     class imbalance) with all three metrics' curves overlaid, plus their
     AUC, alongside the fixed-operating-point tables (open question #6).
   - One shared onset-comparison table (median signed/absolute distance, IQR)
     for the three onset-producing rules against judge onset, since they now
     share the same positive population.
   - Prose comparing the three metrics directly (this is where interleaving
     lost value — now it's a real side-by-side comparison).
   - A closing **"Evaluation metric definitions"** subsubsection: one small
     block of formulas for every metric name used in the tables above
     (Precision, Recall, Specificity, Balanced Accuracy, MCC, Accuracy — the
     same set already listed under "Confusion-metrics helper pattern"),
     stated once so the result tables don't need an inline gloss every time
     one of these names appears. Placed at the end of the unified section
     (definitions after the reader has already seen why they're needed),
     not before.

Rationale for restructuring (vs. keeping interleaved): with a shared protocol,
three near-identical "we evaluate against the judge using population X..."
paragraphs become pure repetition, and scattering the numbers across ~10 pages
makes exactly the cross-metric comparison the document's own intro promises
("none of these metrics is perfect on its own") hard to actually see.

## Notebook changes (code repo)

- All three notebooks now share one calibration/test split (generalized from
  Entropy's existing prompt-hash split) and tune their own threshold
  (repeat-count for LRS, score cutoff for TTR, tau for Entropy) on the
  calibration half by maximizing balanced accuracy, reporting all View
  A/B/C numbers on the held-out test half only.
- `entropy_inspection.ipynb`: rebuild population cells around the new
  definition — negatives from `stop_reason == "eos"` directly (no more join
  against stale `results_*.parquet` rows from the removed stratum).
- `ttr_inspection.ipynb`: extend metrics to Specificity/Bal.Acc/MCC/Accuracy
  now that real negatives exist; keep single-token vs. run-4 comparison; add
  domain-balanced aggregate; add threshold tuning on the calibration split
  instead of the fixed `0.8` cutoff.
- `inspect_dataset.ipynb` (LRS confusion cell): same extension — add real
  negatives, Specificity/Bal.Acc/MCC, domain-balanced aggregate, per-domain
  table with `n`; add threshold tuning (sweep repeat-count cutoff) on the
  calibration split instead of the fixed `>= 3` cutoff; add the per-token
  match flag needed for View B (see above) — this is new, LRS currently only
  emits a rollout-level match count.
- View A addition (all three): threshold-free ROC/PR sweep + AUC, alongside
  the fixed calibrated operating point.
- View B (all three, pending the open-question decision below): token-level
  false-positive rate on EOS rollouts, in-pattern recall on positive
  rollouts past onset.
- View C addition (all three): false early-stop rate on true negatives,
  alongside the existing true-positive offset numbers.
- **Decided 2026-07-22 (see "Notebook architecture" section below): factor
  this into `degeneration_probe/analysis/metric_eval.py` from the start**,
  not as a lower-priority cleanup afterward — build it once against LRS
  first, then have TTR and Entropy call the same functions, instead of
  writing three near-duplicate `confusion_metrics`/`binary_metrics` passes
  and factoring them out later.

## Open questions — resolved 2026-07-22

1. **Minimum-n threshold: resolved to `n_positive(test) < 10`.** Verified
   against the real data (`onset_labels.parquet`, `apertus-8b-instruct`
   build), applying the exact `prompt_split` hash function already used by
   `entropy_inspection.ipynb` (kept as-is, not yet renamed) to every domain:

   | domain | n calib | pos calib | n test | pos test |
   |---|---|---|---|---|
   | aime_2025 | 140 | 2 | 160 | **8** |
   | codeforces | 3170 | 17 | 2830 | **7** |
   | deepmath_103k | 3230 | 98 | 2770 | 62 |
   | if_sft_data_verified | 3230 | 195 | 2770 | 167 |
   | llama_nemotron | 2980 | 58 | 3020 | 65 |
   | medical_o1 | 3010 | 1 | 2990 | **0** |
   | numinamath_1_5 | 2920 | 89 | 3080 | 117 |

   `medical_o1`'s single positive rollout lands entirely in the calibration
   bucket — its test-side positive count is **zero**, not just small; it
   cannot produce a View A/C number on test at all and must be reported as
   "no positive rollouts in test" rather than an insufficient-n row.
   `aime_2025` (8) and `codeforces` (7) both fall under the `n < 10` line;
   the other four domains (58-195) are all comfortably above it. The
   threshold must be checked **on the test-side count specifically**
   (leakage-safe: this is what's actually reported; the total pre-split
   count would understate how thin the reported row really is).
2. **Domain-balancing method: macro-average of per-domain metrics**,
   confirmed — but domains flagged insufficient (medical_o1, and borderline
   aime_2025/codeforces) must be excluded from the macro-average itself, not
   just visually flagged, or they'd distort the "ALL domains" row by
   averaging in a metric computed on 0-8 examples with equal weight to one
   computed on 100+.

   **`medical_o1` treatment, per `docs/analysis/probe_eval_and_training_protocol.md`
   §1.6 (2026-07-22 discussion):** that doc already flags this exact domain
   for the probe's own `test_heldout_domains` evaluation — 1 positive
   rollout out of 6000, "essentially undefined... a single positive example
   carries no statistical weight... should be treated as anecdotal rather
   than a generalization estimate," and states the general rule that "the
   test and validation splits themselves are never resampled or rebalanced,
   since doing so would mean the reported false-positive/recall rates no
   longer estimate what happens on real, naturally-imbalanced generations."
   Applying this directly to our calibration/test split (a second,
   independent split axis from the probe's train/val/test_heldout_domains
   split, used only to keep metric-threshold tuning and reporting disjoint):
   - **Do not** re-hash, re-tag, or otherwise adjust the split to force a
     nonzero `medical_o1` test count — that is exactly the resampling the
     doc rules out, and it wouldn't fix anything anyway: `medical_o1` has
     exactly 1 positive rollout in its *entire* 6000-row population (itself
     already 100% `test_heldout_domains`), so no split assignment can give
     both halves a meaningful count. It's a data-scarcity fact about the
     domain, not a split-design flaw.
   - Report it anyway, explicitly marked, not silently blank: `n = 0`
     positives in test, no computed precision/recall/etc., with a note
     ("0 positive rollouts in test — not evaluable") rather than a `NaN` or
     a 0/0 cell.
   - Exclude it from the domain-balanced macro-average outright (not just
     visually flag it), same treatment as the borderline `aime_2025`/
     `codeforces` rows but for a stronger reason (0 vs. merely low n).
   - `main.tex` must **not** cite `probe_eval_and_training_protocol.md` or
     its "View A/B/C"/"frontier" terminology — restate the anecdotal-only
     caveat for `medical_o1` in self-contained terms local to the unified
     Empirical Performance section.
3. **Section placement: `\subsection`**, confirmed — stays under "2
   Labeling" rather than becoming a new top-level section.
4. **Cost and feasibility wording**: confirmed needs a pass. Additional
   finding while checking domain counts: `results_anthropic.parquet` (890
   rows) already matches `calibration_sample.parquet` (890, `truncated`
   stratum only) exactly — clean. `results_claude_agent_sdk.parquet` still
   has 100 residual `stop_reason == "eos"` judge rows not in the current
   sample (a backup from earlier today had 1244 rows vs. 990 now, so the
   cleanup was partial on that particular backend file). Not a blocker:
   under the new protocol negatives never need a judge verdict at all, so
   these residual rows become irrelevant regardless of whether that backend
   file gets fully cleaned — noted here only so it isn't mistaken for a live
   bug later.
5. **View B sequencing: confirmed — land Views A + C first**, unblocking
   the `main.tex` restructuring; View B (token-level FP rate + in-pattern
   recall, needs a new per-token LRS flag) follows as a second pass.
6. **Threshold-free ROC/PR/AUC: confirmed — include as actual plots in
   `main.tex`**, not just a notebook-only sanity check. One ROC (or PR,
   given the class imbalance — precision/recall tends to be the more
   informative curve here) figure per metric, or one combined figure with
   all three curves overlaid for direct comparison, in the unified
   Empirical Performance section alongside the operating-point tables.

## Correction (2026-07-22, during implementation start): wrong ground-truth source used above

The per-domain positive-count table under open question #1 was built from
the wrong file: `onset_labels.parquet`'s `is_positive` column, which is the
**LRS-based proxy label** used for probe training
(`degeneration_probe/dataset_gen/onset_labels.py`:
`is_positive = stop_reason=="length" AND lrs_normalized_growing onset
resolves`), only *validated* against the judge at 93-99.5% precision, not
the judge verdict itself. Using it as ground truth to evaluate LRS would be
circular. The correct source is `calibration_sample.parquet` (defines the
truncated population, one row per truncated rollout with `domain`/
`stratum`) joined with `results_claude_agent_sdk.parquet`
(`is_degenerating`, `status`, `onset_quote`) — **not**
`results_anthropic.parquet`, which is entirely `status=="failed"` (Anthropic
billing error, "credit balance too low"); a row-count match against
`calibration_sample` (890=890) is not sufficient to call it clean, `status`
must be checked too.

Recomputed real per-domain judge-based numbers (`status=="ok"`,
`is_degenerating==True`, same `prompt_split` hash as before):

| domain | calib pos | test pos | failed (excluded) |
|---|---|---|---|
| aime_2025 | 2 | 8 | 0 |
| codeforces | 18 | 7 | 0 |
| deepmath_103k | 96 | 60 | 0 |
| if_sft_data_verified | 167 | 135 | **62** |
| llama_nemotron | 58 | 65 | 0 |
| medical_o1 | 1 | 0 | 0 |
| numinamath_1_5 | 86 | 116 | 1 |

`medical_o1`'s conclusion (§ open question #2 above) holds exactly: 1
positive total, lands in calibration, 0 in test. `onset_quote` is non-null
for all 819/819 real judge-confirmed positives (better than the assumed
99.6%, though that's before the token-level `find_string_in_tokens` check).
`if_sft_data_verified` has 62 failed judge calls (17% of its truncated
pool) — real gap, not noise. **Decided: proceed now using the 827
successful verdicts, document the 62 failures as "pending retry," do not
block implementation on re-running the judge** (would need to resolve the
`anthropic` backend's billing issue or confirm `claude_agent_sdk` quota
first — out of scope for this pass).

`rollout_signal_df` (built in `inspect_dataset.ipynb` cell 17, reused
across the notebook) already has per-rollout summary scores for **every**
rollout including all `eos` ones — `max_repetition_score` (TTR),
`onset_period_repeat_count` (LRS, digit-normalized-growing), `mean_entropy`
— so the negative side of the new population (`stop_reason=="eos"`) needs
no new computation, just a join against this existing table.

## Notebook architecture: shared module, not one merged notebook (decided 2026-07-22)

User floated consolidating all three metrics' analysis into
`inspect_dataset.ipynb` (as one "all metrics" section) instead of three
separate files, to avoid duplicated code, and asked me to decide based on
how exchangeable the code actually is across metrics. Verdict: **split the
difference — factor the new shared logic into one module, but keep three
notebook homes.**

Reasoning: the *new* protocol logic (population construction from
`onset_labels.parquet`, the calibration/test split + per-metric threshold
tuning, confusion-metric computation, insufficient-n exclusion, the
domain-balanced macro-average, Views A/B/C, ROC/PR/AUC) is **100%
metric-agnostic** — every one of those steps takes a `score`-per-token (or
`is_flagged`-per-rollout) column and `is_positive`/`onset_position` as input
and does not care whether the score came from entropy, TTR, or LRS. Writing
that logic three times, once per notebook, is exactly the kind of
duplication worth avoiding. But each notebook's *existing, already-working*
descriptive/exploratory content (entropy's formula derivation and
worked 0.9/0.1 example, TTR's window-alignment and undefined-label
exploration, LRS's digit-normalization and historical calibration
exploration already living in `inspect_dataset.ipynb`) is genuinely
metric-specific narrative, not something that benefits from being
physically merged into one file — doing so would make an already-large
notebook (`inspect_dataset.ipynb` already covers dataset-wide stats, splits,
and the LRS confusion cell) substantially larger and harder to navigate, for
no shared-code benefit, since that content isn't shared logic to begin with.

Concrete plan:
- New shared module, e.g. `degeneration_probe/analysis/metric_eval.py`,
  exposing one generic evaluation entry point (population construction +
  calibration/test threshold tuning + confusion metrics + domain-balanced
  aggregate + Views A/B/C + ROC/PR/AUC), parameterized by which score column
  and threshold-sweep range to use.
- `entropy_inspection.ipynb`, `ttr_inspection.ipynb`, and the LRS section of
  `inspect_dataset.ipynb` **each keep their own file/section** for
  descriptive content, but call into the shared module for the unified
  Empirical Performance analysis instead of hand-rolling confusion-metric /
  domain-aggregation code locally, as they do today.
- Do **not** move LRS out of `inspect_dataset.ipynb`, and do **not** move
  entropy/TTR into it either — three notebook homes, one shared library.
  This was previously listed as "lower priority than getting the numbers
  right first" (Notebook changes section); given the amount of genuinely
  duplicated new logic, it's worth doing at the same time as the population
  rebuild, not after — building the LRS version straight into the shared
  module (sequencing step 1 below) rather than writing it inline and
  factoring it out later avoids a second pass.

## Suggested sequencing

Non-trivial: touches 3 notebooks + a `main.tex` restructuring across ~400
lines. Do incrementally:

1. **Done (2026-07-22).** Built `degeneration_probe/analysis/metric_eval.py`
   (shared module) and wired it into `inspect_dataset.ipynb`'s LRS cell 30:
   new population, min-n exclusion, calibration/test threshold tuning
   (tuned tau recovers `onset_period_repeat_count >= 3/4/3` across the three
   builds), domain-balanced macro-average, ROC/PR/AUC. Executed end-to-end
   (nbclient) across all three dataset builds, outputs saved in the
   notebook. Also fixed the broken venv (`torch_shm_manager` missing after
   an incomplete resync; `uv sync --reinstall-package torch`) that was
   blocking any notebook execution in this environment.
2. **Done (2026-07-22).** Extended `ttr_inspection.ipynb` the same way,
   while preserving its existing single-token-vs-run-4 comparison and onset
   study (richer than LRS's notebook, not just a confusion-matrix cell): new
   "2.1 Unified population and threshold calibration" section builds the
   unified population and tunes one shared tau per dataset build (on the
   single-token rule, maximizing balanced accuracy on calibration; reused
   for run-4 so the `run4 ⇒ single-token` invariant still holds by
   construction — verified: `run4_only == 0` on all three builds after the
   change). Tuned tau (0.55/0.65/0.60) is notably lower than the old fixed
   0.8. Completion-level tables (Sections 3.1–3.2, 5), the comparison
   section (7), and Section 8's EOS/cap study were all updated to the tuned
   tau and the new metric set; onset study (Sections 4, 6) unchanged in
   population, just reads the tau-updated onset columns. Added a new
   `metric_eval.build_unified_population(..., backend=None)` mode for
   notebooks (like this one) that load a single judge-results file directly
   without a `"backend"` column, and made `roc_pr_curve` drop rows with an
   undefined score (some rollouts are too short for a max-repetition-score)
   instead of erroring. Executed end-to-end, outputs saved in the notebook.
3. **Done (2026-07-22).** Extended `entropy_inspection.ipynb` the same way:
   negative pool swapped from "historically judged natural-EOS rollouts"
   (a now-removed sampling stratum, previously recovered by joining stale
   rows out of `results_*.parquet`) to `stop_reason == "eos"` directly via
   `metric_eval.build_unified_population`. This also removed the old
   `EVALUATED_DATASETS` restriction to 2/3 builds — SFT-256k had no
   historically-flagged EOS rows under the old protocol and was excluded
   from the judge study entirely; under the new protocol negatives need no
   judge call at all, so all three builds are now evaluated. Added Sections
   6.1 (per-domain, min-n exclusion) and 6.2 (ROC/PR/AUC), neither of which
   existed before. Onset study (Section 7) unchanged in population and
   logic (still the truncated, judge-quote-matched population — natural-EOS
   rows are never judged, so they were never part of onset comparison
   anyway). Kept the notebook's own exhaustive `select_balanced_accuracy_threshold`
   sweep (every unique score value) rather than forcing it through
   `metric_eval`'s coarser candidate-grid `tune_threshold`, since Entropy's
   continuous score benefits from the finer sweep and LRS/TTR's own tuning
   approaches already differ from each other for the same reason (discrete
   repeat-count vs. a chosen grid). Executed end-to-end across all three
   builds on the first attempt after the LRS/TTR fixes were already in
   place; ROC AUC (0.70/0.73/0.78) is noticeably lower than TTR/LRS,
   consistent with Entropy's existing characterization as the weaker,
   complementary signal. Outputs saved in the notebook.

   **All three notebooks are now unified.** Positive counts agree exactly
   across LRS/TTR/Entropy for the same build (e.g. 819 for
   apertus-8b-instruct's truncated-judged population), confirming
   `build_unified_population` behaves identically everywhere it's used.
4. Only once all three notebooks agree on final numbers, restructure
   `main.tex`: pull out the three "Empirical performance" subsubsections,
   write the one unified section, recompile with Tectonic
   (`~/.local/bin/tectonic main.tex`, installed this session) and check for
   overfull tables (wrap wide tables in `\resizebox{\textwidth}{!}{...}`, as
   already done for `tab:entropy-performance` and `tab:ttr-falsepos`).
