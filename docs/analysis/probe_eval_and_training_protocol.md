# A unified evaluation protocol and a pluggable training framework for degeneration probes

## 0. Purpose of this note

This note defines, for the onset/degeneration probes:

1. A single **evaluation protocol** for deciding whether one trained probe is
   better than another, and in what specific respect — independent of how
   the probe was trained.
2. A small, config-driven **training framework** so that different training
   recipes can be compared under that same protocol, instead of each recipe
   inventing its own notion of "it works."

This is written for reuse as source material for the paper's evaluation
section — it assumes the reader already knows the labeling story in
`docs/analysis/lrs_and_llm_judge.md` (entropy → TTR → LRS → LLM judge) and the
resulting notion of a **degeneration frontier**: the token position at which a
truncated (`stop_reason == "length"`) rollout starts degenerating, resolved by
`resolve_onset_position()` in `degeneration_probe/dataset_gen/onset_labels.py`.
Everything below calls this position $f_r$ for rollout $r$, and only ever
reads it through that one function, never through a raw LRS/judge field
directly — same discipline the onset-labeling code already enforces.

## 1. The evaluation protocol

### 1.1 The first-alarm position

Every evaluation below is built on one quantity, computed once per test
rollout $r$, per decision threshold $\tau$, and per persistence window $m$
(a positive integer, default $m = 1$):

$$
a_r(\tau, m) = \min\{\, t : p_r(t') \geq \tau \text{ for all } t' \in [t, t+m) \,\}, \quad \text{or } \infty \text{ if no such } t \text{ exists.}
$$

where $p_r(t)$ is the probe's output probability at token $t$ of rollout
$r$. This is the **first-alarm position**: the first token at which the
probe would have fired, at a given threshold, requiring $m$ consecutive
tokens at or above $\tau$ before firing. $m = 1$ recovers plain first-crossing
(fires on a single token above threshold); $m > 1$ trades away some lead time
for robustness against a single noisy spike in $p_r(t)$ — which matters
because a spurious one-token spike is exactly what produces a false
early-stop (§1.4) on an otherwise-negative rollout. $m$ is a property of the
evaluation, not of the probe, and is chosen alongside $\tau$ in §1.5; for
brevity it is written $a_r(\tau)$ everywhere below, with $m$ implicitly fixed
to whatever value §1.5 settles on. It plays the same role here that "first
threshold crossing" plays for TTR and "first match position" plays for LRS
elsewhere in the paper (§ttr-labeling, §lrs-labeling), so probe results slot
into the same table shape already used for those two metrics.

Once $a_r(\tau)$ is computed for every rollout in a split (a single pass over
the probe's per-token predictions), three complementary views are reported
from it, described next.

### 1.2 View A — rollout-level detection

For a fixed $\tau$ (see §1.5 for how $\tau$ is chosen), over all rollouts in
a test split:

- Completion-level confusion matrix: predicted-positive $=[a_r(\tau)<\infty]$
  vs. true `is_positive`. Report precision / recall / accuracy, exactly the
  shape of Tables `ttr-confusion` and `lrs-confusion` in the paper.
- Sweeping $\tau$ gives a rollout-level ROC/PR curve and its AUC, as a
  threshold-free summary alongside the fixed operating point.

### 1.3 View B — full token-level coverage

- **Negative rollouts** (`stop_reason == "eos"`): every single token is
  scored, with no subsampling. The metric is the token-level false-positive
  rate, plus how many rollouts have at least one false positive at all.
- **Positive rollouts**, split at the frontier:
  - $t \geq f_r$ (inside the repeating pattern): recall
    ($1 - $ false-negative rate) is the metric that matters — once inside the
    loop, the probe should catch essentially every token of it. Report
    pooled recall and, per rollout, the count of missed in-pattern tokens
    (not just a rate), since a single missed token deep inside a long loop
    is a very different failure from missing its last 50 tokens.
  - $t < f_r$ (before the frontier): not scored against a positive label —
    these tokens are not degenerate yet — this population feeds View C
    instead.
- Report population sizes alongside every rate (n tokens, n rollouts), so
  the token-level class imbalance between domains/rollouts stays visible
  rather than being collapsed into one blended accuracy number.

### 1.4 View C — early-stopping lead time

- For true positives with $a_r(\tau) < \infty$: distribution of
  $a_r(\tau) - f_r$ (negative = the probe fired before the frontier — lead
  time gained; positive = it fired late). Report median and mean, signed
  and absolute, in the same form as the TTR-vs-judge and LRS-vs-judge onset
  offsets already in the paper (§ttr-labeling, §lrs-labeling).
- For true positives with $a_r(\tau) = \infty$ (missed entirely): report
  separately as "never fired" rather than folding into the offset average.
- For true negatives: any finite $a_r(\tau)$ is a false early-stop — report
  its rate, since it is the direct cost of deploying the probe as an
  early-stopping trigger.

### 1.5 Choosing and freezing $\tau$

$\tau$ is picked once on `val` and frozen before touching `test_indomain` /
`test_heldout_domains`. A reasonable default: pick $\tau$ to hit a target
false-positive rate on `val`'s negative rollouts (a "budget" framing that
matches the early-stopping use case — how much false-alarm cost is
acceptable). "How much false-alarm cost is acceptable" is itself a downstream
deployment decision this note can't settle, so rather than freezing a single
$\tau$, freeze a small family of operating points on `val` (e.g. the
thresholds hitting a 1%, 5%, and 10% negative-rollout false-positive rate)
and report all three views at each, plus the threshold-free rollout-level AUC
from §1.2 as a sanity check that none of them was cherry-picked. The
persistence window $m$ (§1.1) is fixed at $m = 1$ by default across all
operating points — only revisited (swept alongside $\tau$ on `val`) if the
false-early-stop rate at $m = 1$ turns out high enough to matter, rather than
introduced pre-emptively.

### 1.6 Test-split construction

- **Prompt-level composition** (which domains, how many prompts per domain)
  is already stratified — `notebooks/inspect_dataset.ipynb` (Section 5) shows
  each in-domain source (`aime_2025`, `deepmath_103k`, `if_sft_data_verified`,
  `llama_nemotron`, `numinamath_1_5`) contributes essentially the same share
  of `train`, `val`, and `test_indomain` (e.g. `deepmath_103k`/`if_sft`/
  `llama_nemotron`/`numinamath_1_5` are each ~24.7% of every one of the three
  splits; the smaller `aime_2025` domain is ~1.1–1.4% of each). The two
  held-out sources (`codeforces`, `medical_o1`) are 100% in
  `test_heldout_domains` and 0% elsewhere by design, not a stratification gap
  — they exist specifically for zero-shot, cross-domain evaluation.
- **Positive-rollout scarcity in `test_heldout_domains`.** The two held-out
  domains are far from equally positive: in the current
  `apertus-8b-instruct` build, `codeforces` has 24 positive rollouts out of
  6000 (0.4%), while `medical_o1` has exactly **1** positive rollout out of
  6000. Views A and C, reported per held-out domain, are essentially
  undefined for `medical_o1` at this rate — a single positive example carries
  no statistical weight — so `test_heldout_domains` numbers should be read
  per-domain, not pooled, and `medical_o1`'s View A/C numbers specifically
  should be treated as anecdotal rather than a generalization estimate until
  more positive rollouts exist for that domain.
- **Token-level imbalance** (some rollouts contribute thousands of tokens,
  most of them negative) is inherent to the problem. §1.3's rule (report
  rates + population sizes per population, never one blended number) is the
  handling for it — the test and validation splits themselves are never
  resampled or rebalanced, since doing so would mean the reported
  false-positive/recall rates no longer estimate what happens on real,
  naturally-imbalanced generations. **Only the training split is ever
  resampled** (Section 2).

## 2. A pluggable training framework, decoupled from evaluation

### 2.1 Four independent axes

Every training run is a point in a small grid of independent choices, each
its own config field:

1. **`sampling_strategy`** — how (X, Y) examples are drawn from the fixed
   `train` split for one training run (Section 2.2).
2. **`label_source`** — which signal supplies the training target for a
   token (Section 2.3).
3. **`loss_function`** — which loss is optimized against that target
   (Section 2.4).
4. **`lora_scope`** — whether/where LoRA adapts the underlying LM
   (Section 2.6).

Any combination of the four is evaluated with the exact same protocol from
Section 1, so a comparison table (rows = combinations of these axes, columns
= View A/B/C metrics) is the natural output of "which training technique is
better, and why." `label_source` and `loss_function` are related — a
discrete/binary target pairs naturally with a classification loss, a
continuous score with a regression loss — but they are still kept as two
separate fields rather than one combined choice, since a continuous score
can also be thresholded into a binary target and trained with a
classification loss (already effectively what the frontier-derived
multi-horizon labels do internally, turning a distance into a per-horizon
0/1 target). Keeping them separate lets that kind of combination stay
expressible without adding a new named option every time.

### 2.2 `sampling_strategy`: an ablation ladder, not a flat menu

The five `sampling_strategy` options are ordered as a ladder: each rung
changes exactly one design decision relative to the previous one, so that
comparing adjacent rungs under the Section 1 protocol isolates the effect of
that one decision rather than conflating several at once. `label_source`
(`frontier_onset`) and `loss_function` (`bce`) are held fixed across all five
rungs — only `sampling_strategy` varies.

Every rung draws from the same pool: **the entire rollout**, not just the
prefix up to the frontier. A positive rollout's per-token target is the
horizon formula $y_N(t) = 1 \text{ iff } f_r - t \leq N$, applied at every
position $t$ from $0$ to the rollout's last token — for $t > f_r$ this
already evaluates to $1$ for every horizon $N \geq 0$ (the token is inside
the confirmed pattern, so it counts as positive regardless of how far past
the frontier it is), so no separate label rule is needed for the
in-pattern region. Keeping the pool identical across all five rungs is what
makes the adjacent-rung deltas attributable to a single cause: only the
*selection rule* over that pool changes from rung to rung, never the pool
itself.

1. **`full_classification`**: every token of every train rollout, no
   subsampling at all. Class imbalance is handled purely through the loss
   (`pos_weight` in `BCEWithLogitsLoss`, using the corpus-true inverse
   positive rate computed by `onset_dataset.pos_weight_for_horizons`) rather
   than by touching which examples are seen. This is the reference point:
   the whole training population, exactly as generated, with no assumptions
   about what makes an example useful. Also the most expensive and the
   slowest to converge, since most tokens in most rollouts carry no
   information about the frontier.

2. **`rollout_balanced`**: negative rollouts are subsampled to a fixed
   multiple of the positive count (`ImbalancedRolloutSampler`), and, within
   a rollout, a *fixed budget of $W$ tokens* is kept regardless of the
   rollout's true length — $W$ chosen equal to the window size used by every
   later rung, so token budget per rollout is held constant from here on and
   is never again a confound between rungs. The $W$ kept tokens are a
   uniform random sample over the rollout's *entire* length (positive or
   negative alike — a positive rollout can contribute post-frontier,
   deep-in-pattern tokens here just as freely as pre-frontier ones). This
   isolates one question relative to rung 1: *does correcting the
   rollout-level class ratio, on its own, help — independent of which
   specific tokens end up in the fixed budget?*

3. **`random_window`**: same fixed budget as rung 2 ($W$ tokens per
   rollout), but the $W$ tokens must now be *contiguous* — one randomly
   placed window of length $W$ anywhere in the rollout (again, the full
   length, not restricted to the pre-frontier region), instead of $W$ tokens
   sampled independently. This isolates: *does replacing an
   independently-sampled token budget with one contiguous window change
   anything*, holding the window's position unconstrained (still random)?

4. **`frontier_window`**: same contiguous window of length $W$, drawn from
   the same full-rollout pool as rung 3, but for positive rollouts the
   window's position is no longer random — it is anchored on the frontier
   $f_r$, with $W$ chosen $\geq$ the longest horizon of interest so every
   horizon's positive and negative tokens stay representable inside one
   window. This is a deliberate narrowing relative to rung 3 — the window
   can no longer land just anywhere, only near the frontier — which is
   exactly the comparison of interest: *does anchoring the window
   specifically on the frontier, rather than placing it anywhere in the
   rollout, help?* Negative-rollout windows stay randomly placed, unchanged
   from rung 3 (there is no frontier to anchor to in a negative rollout).
   `frontier_window` is also the rung with the best cost/generalization
   property independent of the ablation logic: every rollout, old or new,
   contributes exactly one window of fixed size, so adding a new domain or a
   much longer rollout never changes per-example cost or the effective
   class balance.

   *Anchor style* is itself a second, open experimental choice within this
   rung, independent of the rung-3-vs-4 comparison above: a `trailing`
   window (the $W$ tokens ending at $f_r$, entirely pre-frontier/lead-up)
   vs. a `centered` window (roughly $W/2$ tokens on each side of $f_r$,
   mixing lead-up tokens with confirmed in-pattern ones). `trailing` matches
   the early-stopping deployment framing directly — it's exactly the
   context a live decision would have, nothing past "now" — and keeps every
   token in the window close to the ambiguous boundary. `centered` also
   exposes training to clearly-positive, already-in-pattern tokens in the
   same window, which could stabilize the positive class's representation
   but spends part of the fixed budget $W$ on the easier, already-degenerate
   region rather than the harder boundary one. Neither is obviously better a
   priori — this is deliberately left as a value to sweep (`trailing` as the
   default to compare `random_window` against, `centered` as a second point
   once that comparison is in), not a decision made here.

5. **`frontier_window` + hard-negative mining**: identical to rung 4 for
   positive rollouts; for negative rollouts, the window is no longer placed
   uniformly at random. It is instead biased toward spans with an elevated
   TTR-based repetition score or an LRS match (§ttr-labeling,
   §lrs-labeling) — i.e. spans that look structurally repetitive by the same
   heuristics used earlier in the labeling pipeline, yet belong to a
   confirmed-negative rollout, which is exactly the shape of the confusable
   cases the LLM judge exists to rule out (§llm-judge-labeling: genuine
   incremental work, instructed repetition). A configurable mix ratio
   (e.g. a fraction of windows drawn from the hard-negative pool, the rest
   uniformly at random as in rung 4, for coverage/calibration) is a
   parameter of this rung, not a separate one. Isolated question: *does
   deliberately exposing the probe to structurally-repetitive-but-legitimate
   negatives, instead of drawing negatives uniformly at random, reduce
   false positives specifically on the kind of text those metrics are known
   to misfire on?*

All five rungs share one call signature — given the fixed `rollout_index`
DataFrame (already carrying `domain`, `prompt_id`, `is_positive`,
`onset_position` from `onset_labels.py`) plus a layer, produce `(X, Y)` or a
batch sampler — so `TrainingConfig` gains a
`sampling_strategy: Literal["full_classification", "rollout_balanced", "random_window", "frontier_window", "frontier_window_hard_negative"]`
field that dispatches to one of five functions in (e.g.) a new
`degeneration_probe/data/sampling_strategies.py`, mirroring the dispatch
`ProbeTrainer.compute_loss` already does on `task`. Rungs 2–5 all keep
reusing the existing `OnsetActivationDataset`/`materialize_features`
machinery — only the *index* (and, for rung 5, the window-selection weights)
each rung hands to it differs.

#### 2.2.1 Choosing $W$

$W$ is not fixed a priori. Three candidate values — $64$, $128$, $256$ — are
piloted on rungs 1–2 only (the cheapest pair in the ladder) before committing
to a single $W$ for rungs 3–5; running the full five-rung ladder at all three
values would triple the ladder's cost to answer a question (window size)
that is orthogonal to what the ladder itself is measuring (which selection
rule helps). Whichever $W$ wins the pilot is then locked for the remaining
rungs, consistent with §2.5's preference for evidence-driven parameter
choices over pre-emptive sweeps. Rollouts shorter than $W$ contribute every
eligible token they have — no padding, no exclusion — since falling short of
$W$ is a property of the rollout, not a reason to distort or drop it.

#### 2.2.2 Run-to-run variance across seeds

Every rung (and, during the $W$ pilot, every candidate $W$) is run at a small
fixed set of seeds — reusing the single `TrainingConfig.seed` field
(`config.py:140`) that already drives probe/LoRA weight init (`setup_probe`),
`torch.manual_seed`, and the sampling randomness itself
(`ImbalancedRolloutSampler`, negative-token subsampling in
`OnsetActivationDataset`), rather than introducing a separate seed per
randomness source. Report mean $\pm$ std of every View A/B/C metric across
those seed repeats, so an adjacent-rung delta (§4) is judged against its
noise floor rather than a single point estimate. This conflates
training-dynamics noise with sampling-strategy noise into one combined
variance; that is an acceptable simplification for the ladder itself, since
decomposing the two would require new seed plumbing with no evidence yet that
it's needed.

### 2.3 `label_source`: which signal defines the training target

A token's training target can come from more than one signal, and the choice
is independent of `sampling_strategy`:

- **`frontier_onset`** (current default): the discrete, per-horizon 0/1
  target derived from the degeneration frontier $f_r$ via
  `resolve_onset_position()` and the horizon encoding in
  `onset_dataset.OnsetActivationDataset._labels_for` — `y_N(t) = 1` iff
  $f_r - t \leq N$. This is the target the onset probes (`probe_N` /
  multi-horizon) train against today.
- **`repetition_score`**: the continuous TTR-based repetition score
  (§ttr-labeling) at each token, already used as a regression target by the
  existing `task="repetition_score"` path (`training/trainer.py`,
  `compute_probe_regression_loss`).
- any future signal (e.g. the LLM judge's `onset_quote`, once populated at
  scale) — `resolve_onset_position()` already treats this as a swappable
  `onset_metric`, so a new frontier-defining signal is a one-line addition
  there and does not require a new `label_source`; a new `label_source` is
  only needed for a signal that isn't a frontier position at all (e.g. a
  continuous per-token score with no single onset point).

Every `label_source` exposes the same shape of output — a per-token target
(plus an ignore-mask for positions with no defined label) — so
`sampling_strategy` (Section 2.2) stays agnostic to which one is in use: it
selects *which tokens* go into a batch, `label_source` decides *what value*
each selected token is trained toward.

### 2.4 `loss_function`: which loss is optimized

- **`bce`** (binary cross-entropy, optionally per-horizon and weighted):
  `compute_probe_bce_loss` / `compute_multi_horizon_bce_loss` in
  `training/loss.py`. The natural pairing for `frontier_onset`'s discrete
  horizon targets.
- **`mse` / `smooth_l1` / `l1`**: `compute_probe_regression_loss`, already
  implemented and selected via `TrainingConfig.regression_loss`. The natural
  pairing for a continuous `label_source` like `repetition_score`.

`loss_function` is what today is implicitly fixed by the `task` literal
(`"repetition_score"` → a regression loss, `"onset_multi_horizon"` →
multi-horizon BCE). Exposing it as its own field alongside `label_source`
means, for instance, the frontier-derived horizon targets could be trained
with a regression loss against the raw distance-to-onset value instead of a
thresholded 0/1 — a combination the current `task` enum cannot express
because it bundles label and loss into one name.

### 2.5 Cross-cutting stratification (domain / prompt)

Stratification by domain and prompt is orthogonal to which rung of the
ladder above is chosen, and is an additional, optional parameter of
whichever sampler underlies that rung, since `rollout_index` already
carries the needed columns:

- `stratify_by_domain: bool` — when subsampling negatives (rung 2) or
  drawing negative windows (rungs 3–5), draw proportionally to each domain's
  share of the pool rather than uniformly at random over all negative
  rollouts pooled together. This also interacts directly with rung 5's
  hard-negative pool, since the confusable, structurally-repetitive-but-legitimate
  cases are themselves domain-specific (brute-force enumeration concentrates
  in the math/code domains, instructed repetition in `if_sft_data_verified`)
  — domain-proportional sampling and hard-negative mining should be thought
  of as one combined selection rule for rung 5, not two independent filters
  applied in sequence.
- `max_windows_per_prompt: Optional[int]` — a cap on how many of a batch's
  positive examples may come from the same `prompt_id` (relevant mainly to
  `frontier_window`, where a prompt with many sampled rollouts could
  otherwise flood a batch with near-duplicate windows).

Recommendation: implement domain-proportional sampling now (it's nearly
free, the column already exists), but treat the prompt-level cap as a
stretch goal, added only if cross-prompt overfitting is actually observed
empirically (e.g. a gap between `test_indomain` and `test_heldout_domains`
in Section 1's metrics that domain-proportional sampling alone doesn't
close) — see [[project_direction]] for the same preference expressed for
window-score parameter selection: prioritize interpretability and evidence
over adding knobs pre-emptively.

### 2.6 `lora_scope`

`ProbeConfig.lora_layers` accepts `"all"` (resolves to every layer from 0 up
to the probed layer, `config.py:75-84`), `"none"` (frozen model), or an
explicit list of layer indices. Subtasks G–K of the onset-probes project
already validated this end-to-end (LoRA vs. no-LoRA comparison at layer 16).
The remaining work is to treat `lora_scope` as an explicit axis in the same
results table as the other three — comparisons so far have only varied it at
one fixed, implicit sampling/label/loss regime.

## 3. Implementation status

### 3.1 Already implemented (reuse, don't rebuild)
- Degeneration frontier / onset position: `resolve_onset_position()`,
  `degeneration_probe/dataset_gen/onset_labels.py`.
- The rollout-population-ratio half of rung 2: `ImbalancedRolloutSampler`
  (negative rollouts subsampled to a configurable multiple of the positive
  count) in `degeneration_probe/data/onset_dataset.py`, plus a per-rollout
  negative-token cap (`max_negative_tokens_per_rollout`) in the same module —
  reusable as-is for rung 2's negative side. The positive side still needs
  the fixed-budget change described in §3.2.
- Corpus-true `pos_weight` for loss-based rebalancing (used across every
  rung, not just rung 1): `onset_dataset.pos_weight_for_horizons`.
- `lora_scope` axis: `ProbeConfig.lora_layers`, validated in subtasks G–K.
- Every `loss_function` option (`bce`, `mse`/`smooth_l1`/`l1`):
  `training/loss.py`
  (`compute_probe_bce_loss`/`compute_multi_horizon_bce_loss`,
  `compute_probe_regression_loss`).
- Both `label_source` options (`frontier_onset`, `repetition_score`): the
  former via `resolve_onset_position()` + the horizon encoding in
  `onset_dataset.py`, the latter via the existing `task="repetition_score"`
  path.
- Prompt-level domain-stratified train/val/test splits (see
  `notebooks/inspect_dataset.ipynb`, Section 5) — no change needed here.

### 3.2 Not yet implemented
- **Reading past the frontier at all.** `OnsetActivationDataset` currently
  caps a positive rollout's read at `onset_position + 1` — every position
  after the frontier is never loaded, for training or evaluation. Both the
  full-rollout pool every rung in §2.2 now relies on, and View B's
  in-pattern recall check ($t \geq f_r$, §1.3), need those positions
  available. This is a change to the underlying activation read, not just
  to which positions get selected afterward — concretely, the fix is
  localized to the `eligible_end` computation in
  `OnsetActivationDataset.__getitem__` (`onset_dataset.py:132`), which today
  reads `onset_position + 1` for positive rows and should read
  `num_tokens` instead, matching the negative-row branch. No change is
  needed to the label formula itself: `_labels_for`'s
  `distance = onset - t` already goes negative for $t > f_r$, and
  `distance <= horizon` still correctly evaluates to `True` there, so
  positions past the frontier are already labeled correctly once they're
  read at all.
- **Full, unsampled negative-token coverage at eval time.**
  `scripts/evaluate_onset_probes.py` currently caps negative-token reads per
  rollout (`EVAL_NEGATIVE_TOKENS_PER_ROLLOUT = 32`) for `val`,
  `test_indomain`, and `test_heldout_domains` alike. View B (§1.3) requires
  every token of every negative rollout, since any cap can only
  underestimate the false-positive rate. Checked against the current
  `apertus-8b-instruct` build: negative rollouts across the three eval splits
  total ~8.6M tokens uncapped vs. ~0.6M at the current 32-token cap (~14x
  more reads), which at `HIDDEN_SIZE = 4096` (one layer,
  `onset_dataset.py:56`) is roughly 130GB of additional activation reads
  across a full eval pass. That's a real increase in eval I/O and wall-clock
  time, but not prohibitive — no fallback partial cap is needed.
- **A thresholded operating point.** The current pipeline only reports AUC /
  AP / Brier (rank- and calibration-based, threshold-free). Views A and C,
  and the confusion matrix/precision/recall/F1 they produce, all require
  picking and freezing a $\tau$ (§1.5), which nothing currently does.
- **Rung 2's fixed token budget.** Today a positive rollout always
  contributes its *entire* eligible range (currently truncated at $f_r$, see
  the read-past-the-frontier item above), not a capped $W$-token sample —
  rung 2 as defined in §2.2 needs this capped, both so its token budget
  matches rungs 3–5's window size and so it actually isolates
  "population-level rebalancing" without also changing "how much of each
  rollout is seen."
- **Rungs 3, 4, 5 (`random_window`, `frontier_window`,
  `frontier_window` + hard-negative mining)**: none built yet.
- **`sampling_strategy` as an explicit, named config field**: doesn't exist
  yet; today there is exactly one (implicit) strategy, closest to an
  unbudgeted version of rung 2.
- **Domain-proportional negative sampling**: not implemented; today
  `ImbalancedRolloutSampler` draws negatives uniformly over the whole pool
  regardless of domain.
- **Hard-negative selection signal for rung 5**: not implemented; would read
  the existing per-token repetition score and LRS match fields already
  computed by the labeling pipeline (`degeneration_probe/dataset_gen/label.py`)
  for negative (`stop_reason == "eos"`) rollouts, to bias window placement
  toward locally repetitive-but-legitimate spans.
- **`label_source` and `loss_function` as explicit, independent config
  fields**: each option they'd select between already exists, but bundled
  implicitly inside the `task` literal (`"hallucination"`,
  `"repetition_score"`, `"onset_multi_horizon"`) rather than exposed
  separately — so today `label_source` and `loss_function` cannot be
  varied independently of one another.

## 4. Suggested next steps

The first training runs target the current single-horizon setup (`task =
"hallucination"`, i.e. `probe_N`) as the baseline — `onset_multi_horizon` and
how §1's views generalize to a per-horizon probe output are deferred and do
not block anything below.

1. Fix `OnsetActivationDataset` to read past the frontier (§3.2) — change
   `eligible_end` for positive rows from `onset_position + 1` to
   `num_tokens`. This is a prerequisite for every step below: View B's
   in-pattern recall and every rung's full-rollout pool both depend on it.
2. Extend the eval harness to score every token of every negative rollout
   (§3.2) — this determines whether any false-positive-rate number reported
   so far can be trusted.
3. Pick and freeze a small family of $\tau$ operating points on `val` (§1.5),
   plus the persistence window $m$ (§1.1, default $m = 1$), then implement
   views A/B/C as one shared reporting function so every future probe
   (regardless of training recipe) is scored the same way.
4. Add `sampling_strategy` as an explicit config field. Pilot
   $W \in \{64, 128, 256\}$ on rungs 1–2 only and lock the winner (§2.2.1),
   then implement the rest of the ladder in order — each rung is a small
   diff on the previous one, so building them in sequence (rung 1 → 2 → 3 →
   4 → 5) doubles as the ablation itself: cap rung 2's positive-side budget
   to the locked $W$ first (§3.2), then rung 3 (contiguous window, random
   placement), then rung 4 (frontier-anchored placement), then rung 5
   (hard-negative window selection for the negative side).
5. Run all five rungs, at a fixed `lora_scope`, through the Section 1
   protocol, each rung at $\geq 3$ seeds (§2.2.2), and report the
   adjacent-rung deltas (1→2, 2→3, 3→4, 4→5) as mean $\pm$ std — each delta
   attributable to exactly one design decision. Within rung 4 (and by
   inheritance rung 5), also sweep `trailing` vs. `centered` anchor style
   (§2.2) as a second, independent comparison once the main ladder result is
   in.
6. Split `label_source` and `loss_function` out of the `task` literal into
   their own config fields, so existing combinations (`frontier_onset`+`bce`,
   `repetition_score`+regression) keep working unchanged while new
   combinations become expressible.
7. Cross the winning `sampling_strategy` rung with `label_source`,
   `loss_function`, and `lora_scope` to fill in the full comparison table
   from §2.1.
