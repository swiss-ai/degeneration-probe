# Prompt for the writing-assistant session

Paste everything below into a new Claude Code chat opened in this repository.

---

You are my writing assistant for a NeurIPS submission about detecting
repetitive degeneration in language model generations from hidden states. Your
job has three parts:

1. **Verify before you write.** Every number, every claim about our own system,
   and every claim about another paper must be checked. Our numbers come from
   the code and the run directories on disk. Claims about the literature come
   from the primary source online. Never carry a number from one of our own
   documents into the paper without re-deriving it.
2. **Apply the agreed edits** to `docs/paper/main.tex`, listed below.
3. **Populate `notebooks/paper.py`** with every table and figure the paper
   needs, saving each figure as a PDF for inclusion.

---

## Task 0, before anything else: audit the references

**This is the single most important thing to do, and it comes first.**

`docs/paper/main.tex` and `docs/paper/references.bib` make claims about other
people's work. Some were checked against primary sources, some were carried over
from an AI-written literature review and have never been checked. Every entry in
`references.bib` carries a `[checked]`, `[carried]` or `[partial]` mark saying
which.

For **every** cited work, confirm from the paper itself, not from a summary:

- the metadata: title, authors, venue, year, pages
- every number I attribute to it
- **at what level it measures**: per token, per sentence or chunk, or per whole
  answer. I make claims about this and they are load-bearing.
- **which layer or representation it probes**. I claim the nearest works read
  the final layer, and an entire section of the paper depends on that being
  true.
- which decoding it uses, and how its evaluation set is composed

Specific claims to confirm or refute, all currently unverified:

- Duan et al. 2026: early detection rate 0.64--0.76, false positive rate
  0.24--0.34, lead times of roughly 1300--2000 tokens, classifiers on averaged
  final-layer hidden states reaching accuracy and AUC above 0.99, greedy
  decoding, balanced test sets of at least 50 positive and 50 negative cases.
- Xie et al. 2025: roughly 90--94% accuracy and 96--99 AUROC at temperature
  zero, a linear classifier on the final block's hidden state at the
  chunk-final token, chunks labelled by embedding similarity at 0.99. Also
  confirm the appendix example in which the classifier score rises from
  near zero to certainty hundreds of chunks into the loop, since I use it.
- Yu et al. 2025: 95.24% accuracy, F1 0.87, false positive rate 2.59%, whether
  the detector is answer-level, and whether a warm-up window is used.
- LoopGuard 2026: the exact loop label thresholds, the online trigger, and its
  sliding window size.
- Kramár et al. 2026: the context-length shift findings.
- Everything marked `[carried]` in `references.bib`.

Report back a table of what held, what did not, and what you could not access.
Where a claim fails, tell me before rewriting the sentence around it.

---

## The other important instruction

**Several documents in this repository were written by an AI and contain
mistakes.** Treat all of them as leads, never as evidence:

- `docs/analysis/labeling-strategy-analysis.tex`
- `docs/analysis/related_work_metrics.md`
- `docs/analysis/probe_eval_and_training_protocol.md`
- `docs/contribution_and_positioning.md`, `docs/paper_draft.md`
- `notebooks/experiment_diary.py`, `notebooks/lora_linearity_probe.py`
- `docs/paper/main.tex` itself

Known failures, so you recognise the pattern: one document states the label
horizon has no effect when the runs show its warning coverage roughly doubling,
monotonically, in every family; another cites Hewitt and Liang to the wrong
venue; another reports a cross-model transfer result computed on a superseded
dataset build without saying so.

When a document and the artifacts disagree, the artifacts win, and you tell me
the document was wrong rather than quietly using the better number.

---

## Where ground truth lives

**Code.** `degeneration_probe/`. Most relevant: `evaluation/protocol.py`
(thresholds, first alarm, persistence, the four views, coverage windows),
`evaluation/head_selection.py` (the checkpoint and depth rule),
`dataset_gen/label.py` (structural metrics), `dataset_gen/llm_judge.py` (the
judge and its verification), `data/sampling.py` and `data/windowed_dataset.py`
(token selection).

**Results.** `outputs/<run_name>/<attempt>/`: `run_info.json` (identity, axes,
status, git commit), `resolved_config.json`, `checkpoint_replay.parquet` (every
saved checkpoint re-measured at every depth on a pinned population),
`layers/layer_NN/scores/<split>.parquet` (one score per token),
`layers/layer_NN/evaluation/<split>/*.csv` (four views, a per-rollout table, a
per-domain table), `sweep/layer_NN/<checkpoint>/` for other checkpoints.
`outputs/baselines/{repetition,rep_l,lrs,entropy}/` holds model-free scorers in
the same shape.

**Corpus.** `/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/`,
one directory per build. `onset_labels/onset_labels.parquet` is the label table
training reads: `split`, `domain`, `stop_reason`, `is_positive`,
`onset_position`, `onset_resolution`. `labels/<domain>/*.parquet` holds
per-token structural metrics. `generations/<domain>/*.parquet` holds generated
token ids and text.

**Provenance.** Scores directories now carry `scoring_provenance.json` naming
the checkpoint that filled them. Older ones do not and their checkpoint cannot
be recovered; `scripts/audit_score_provenance.py` says which are which. Never
attribute a number to a checkpoint whose directory has no provenance file
without saying it is unattributed.

---

## Facts that are true and easy to get wrong

- **The model is public.** Apertus-8B-Instruct-2509 is Apache 2.0 with open
  weights and open data. Only the two Apertus 1.5 checkpoints, used in one
  transfer row, are unreleased.
- **The windowed repetition score is not ours.** It is `seq-rep-2` on a moving
  window. Cite both sources rather than defending an invention.
- **Its window looks forward.** Correct for annotating a finished answer, wrong
  for a scorer a live monitor could run. `rep/l` is the literature's natively
  backward-looking per-token signal and is already scored as a second baseline.
  **Do not describe the forward window as a mistake in the literature. Nobody
  else made it.**
- **Test splits are read once.** `test_indomain` has never been scored for any
  probe. Do not touch it without asking.
- **Two workstreams are in flight**: probes trained directly on the Llama and
  Mistral builds, and every LoRA run repeated under a corrected stopping rule.
  Leave placeholders; do not use the old LoRA numbers.
- **Persistence has never been used.** Everything on disk is `m = 1`.

---

## Task 1: edits to `docs/paper/main.tex`

Find the open questions with `grep -n 'NOTE:' docs/paper/main.tex`. All 26 are
answered and approved. Apply them, then delete each `[NOTE: ...]` marker.

**Wording and clarity**

1. Line 64: define "matched null" concretely at first use in the controls
   section, and in the contributions say "compared against healthy tokens
   matched on that variable".
2. Line 74: replace "the metric this literature reports" with "answer-level
   detection accuracy".
3. Line 90: replace "linear readout" with "a linear probe reading one layer's
   hidden state".
4. Line 102: remove the `rep/l` sentence from Related Work. Introduce `rep/l`
   in the baselines section, immediately after the forward-window problem is
   stated.
5. Line 309: the positive weight is computed from the class balance of the
   tokens that reach the loss after selection, not from the raw corpus. Say
   exactly that.
6. Line 358: "cannot be settled here" means the acceptable false-alarm cost
   depends on a deployment we do not have, so several budgets are reported
   instead of one being chosen.
7. Line 360: rewrite with the mechanism and the numbers. When many healthy
   answers tie at a scorer's maximum, no threshold isolates a small share of
   them; the threshold is pushed above the range and nothing fires at all,
   positives included, so the row reads as zeros that mean the opposite of
   silence. Verify and quote the tie fractions for LRS, entropy and `rep/l`.
8. Line 379: rename "healthy token rate" to "token-level false-alarm rate on
   healthy answers", and say how it differs from the answer-level budget.
9. Line 382: lead time is a **median**, over degenerate answers **that fired**,
   with never-fired counted separately.
10. Line 703: change to `\citet{page1954continuous}` and introduce it as the
    origin of the cumulative-sum change detector, matching how other works are
    introduced.

**Structure**

11. Line 327: move the checkpoint-and-depth subsection to **after** the
    evaluation protocol section. It is expressed in evaluation quantities and
    currently uses "coverage" and "run-up" before either is defined.
12. Line 353 and line 366: **reorder so the first alarm is defined before the
    operating point.** Then state the chain explicitly: score per token, first
    alarm `a_r(tau, m)`, an answer is predicted positive iff its first alarm is
    finite, the false-alarm rate is the share of healthy answers with a finite
    alarm, and tau is solved so that share equals the budget at fixed `m`. At
    `m = 1` this reduces to "the maximum token score reaches tau". Use the
    first-alarm form as the definition, since it generalises to persistence.
13. Line 372: define detection as the answer-level confusion matrix built from
    that rule, and list what is reported: recall, precision, AUC and average
    precision.
14. Line 341: cut the scores-as-interface paragraph to one sentence, the one
    saying it makes probes and baselines comparable by construction. Move the
    rest to the appendix.
15. Line 347: remove the synthetic-scorer validation from the main text; one
    clause in the appendix.

**Scope and honesty**

16. Line 178: state that the held-out code source and the in-domain code source
    are **both code**, so the held-out result tests generalisation across
    sources within code rather than across kinds of text. Restate it that way
    wherever the held-out result appears.
17. Line 188: keep "treating every naturally terminated answer as a negative is
    an assumption". Add a Limitations paragraph saying no naturally terminated
    answer has ever been judged, so an answer that began degenerating and then
    happened to emit an end-of-sequence token would be silently mislabelled.
    Add a `\TBD` marking the experiment as still to be done: judge a stratified
    sample of high-repetition naturally terminated answers and report the
    contamination rate.
18. Line 224: keep the corpus table on the main build; move all builds to an
    appendix table.
19. Line 295: add a `\TBD` saying entropy is supported as a token-level target
    and has not been run, and state that the longest repeated substring is not a
    per-token score and would in any case be a noisier version of the judge's
    own label.
20. Line 385: see task 2 below; either the histogram supports the claim or the
    claim changes to what is measured.
21. Line 392: replace "trivially separable" with what is measured. See task 2.
22. Line 267: add a short appendix subsection arguing the multi-head
    equivalence: summing independent per-head losses gives each head the
    gradient it would receive alone, because no parameter is shared, provided
    the gradient norm is clipped per head and each head keeps its own
    normalisation.

---

## Task 2: experiments to run, in priority order

**A. Alternative selection metrics (highest value).** The paper claims
answer-level metrics are saturated. Make it concrete: for the same runs, report
which checkpoint is selected when selecting on accuracy and on AUC, beside what
the current rule selects, and what each choice costs in coverage of the run-up.
The replay does **not** store AUC, so recompute from the stored per-token
scores. No retraining, no GPU.

**B. Activation self-similarity baseline.** A model-internal scorer that needs
no training: the cosine similarity between the current hidden state and recent
previous ones, at the reported layer. This is close to what Yu et al. use, so it
is a literature baseline rather than a straw man, and it answers the sharpest
question about the probe: does a trained probe beat simply noticing that the
residual stream has started repeating itself? Score it through the existing
score-file interface. Activations are cached, so this is I/O bound. Confirm the
exact form Yu et al. use before naming it after them.

**C. Persistence sweep.** Already implemented: `evaluate_scores.py` takes
`--compare-persistence`. Run it over stored scores at several `m`, report in the
appendix, and keep `m = 1` in the main text as a stated choice rather than an
unstated assumption.

**D. Alarm-offset histogram.** The distribution behind the median offset, which
appears in nearly every table. Draw it before writing the sentence it supports:
for one representative candidate at the 1% budget the offsets are roughly 20
answers more than 100 tokens early, 7 in the last 100 before the frontier, 26
within 100 after, and 55 more than 100 after. That is heavily skewed with a dip,
and **"bimodal" may be too strong**. Default to placing it in the main text,
because a headline statistic whose distribution is never shown invites the
question; move it to the appendix only if space forces it.

---

## Task 3: populate `notebooks/paper.py`

This marimo notebook generates everything the paper prints, so that no number is
typed by hand. It currently builds three summary tables from the run
directories.

Read `~/.claude/prompts/marimo.md` before editing it. Every figure must be saved
as a PDF as well as rendered. Reuse the existing colour and style helpers rather
than inventing new ones.

**Tables**, in paper order: corpus populations; model-free baselines at fixed
budgets; the depth profile; token selection; label family and horizon; the
position-matched decomposition; per domain including the held-out code source;
the protocol bridge ladder; transfer.

**Figures**: a task-definition figure showing one answer with its frontier,
probe score and threshold; a **selection-strategy diagram**, showing a few
rollouts as strips of token squares with the frontier as a vertical line and the
tokens each strategy trains on highlighted, one row per strategy; the depth
profile; the confound decomposition; the per-domain profile; the alarm-offset
histogram.

---

## How to work

- Compute-heavy work goes through `sbatch`; this is a shared login node. Cheap
  analysis, `evaluate_scores.py`, `pytest` and file inspection run locally.
- `.venv/bin/python -m pytest tests -q` is the fastest check on a change.
- Show me the numbers you derived and the command that derived them, so I can
  catch a wrong population or a wrong split.
- Tell me when a claim cannot be verified rather than softening it into
  something defensible.

## Writing style

Write as if I am the sole author describing the system as it is now. No
narration of what changed or of our conversation. Plain English, short
sentences. No em dashes inside sentences. Never add a Claude co-author line to a
commit.
