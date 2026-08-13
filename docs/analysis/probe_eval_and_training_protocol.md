# Degeneration probes: design, training and evaluation protocol

## 1. What this system is

A degeneration probe is a small classifier that reads one transformer layer's
hidden state at every generated token and emits a probability in $[0, 1]$ that
the generation has entered, or is about to enter, a degenerate repetition loop.
The intended use is early stopping: a live generation is scored token by token,
and once the probe is confident the rollout is looping, generation can be cut
short instead of running to the token limit.

The system has two halves, kept deliberately separate:

- A **training framework** in which several recipes (which tokens are trained
  on, what value each token is trained toward, which loss, whether the
  underlying model adapts) can be expressed as configuration rather than as
  separate code paths.
- An **evaluation protocol** that judges any per-token scorer without knowing
  how it was produced. The same protocol scores a trained probe, a probe
  trained with a completely different recipe, and a non-learned baseline such
  as a repetition heuristic.

The separation is the central design choice. A recipe is only interesting if it
wins under a protocol that was not designed around it, and the protocol changes
far more often than the training code does, so the two are never allowed to
share state beyond a single, well specified file of scores.

## 2. The corpus

### 2.1 Rollouts, labels and the degeneration frontier

The corpus is a set of prompts drawn from several source datasets, each
completed several times by the target model under sampling. Every completion
(a *rollout*) is stored with its exact sampled token ids, never with re-encoded
text, so that everything computed downstream stays aligned to the same token
indices.

A rollout is **negative** when generation ended naturally at an end-of-sequence
token. A rollout is **positive** when generation was cut off by the token limit
*and* a degeneration onset can be resolved for it. Truncated rollouts for which
no onset can be resolved are excluded from training and evaluation entirely,
rather than being given an invented onset or quietly treated as negative.

For each positive rollout $r$ there is one **degeneration frontier** $f_r$: the
token position at which the rollout starts repeating. A single component owns
the definition of $f_r$ and every other part of the system reads the frontier
only through it, so the underlying signal stays swappable.

The frontier comes from an LLM judge, which reads a rollout and names where it
begins to degenerate by quoting the text. Asking for a quote rather than a
number is what makes the label checkable: a quote can be verified against the
completion it came from, and a person reading the rollout can see immediately
whether the judge pointed at the right place. A quote is not the form training
needs, so a separate step locates it in the rollout's own token stream and
records the index of its first token. That resolution is cached, since it is
the one expensive part of the chain and changes only when the judge is re-run.
A quote that cannot be located leaves its rollout excluded, with the reason
recorded rather than the rollout silently dropped.

The corpus also carries per-token repetition and entropy measures, and a
whole-rollout longest-repeated-substring match. These are scored through the
same evaluation protocol as the probe, as the baselines it has to beat, and
they are what the judge's reliability is checked against. They play no part in
deciding where a rollout starts degenerating.

Two asymmetries of the corpus shape almost every decision that follows:

- **Positive rollouts are long and mostly degenerate.** Because a positive
  rollout is by definition one that hit the token limit, each one contributes a
  short pre-frontier region and a much longer in-pattern tail. Negative
  rollouts are far shorter on average, since they stopped when the model was
  done.
- **Imbalance points in opposite directions at the two levels.** Counted as
  rollouts, negatives outnumber positives by more than an order of magnitude.
  Counted as tokens, the in-pattern tails of the positives are large enough
  that the two classes come close to parity once negative rollouts are
  subsampled at all. Any design that treats "the positive class is rare" as a
  single fact will over-correct.

### 2.2 Splits

Splits are assigned at the **prompt** level, so all rollouts of a prompt share
one split. This is enforced, not assumed: rollouts of the same prompt are near
duplicates of one another, and a prompt straddling two splits would inflate
every in-domain number.

There are four splits:

- `train`, `val` and `test_indomain` are drawn from the same set of source
  domains and are stratified so that each domain contributes about the same
  share of each of the three.
- `test_heldout_domains` consists of domains that appear nowhere else. It exists
  for zero-shot, cross-domain measurement, so its absence from training is by
  design and not a stratification gap.

Held-out domains differ sharply in how often they degenerate at all, and at
least one of them yields only a handful of positive rollouts in total. Held-out
results are therefore always reported per domain, never pooled, and the
reporting code marks any per-domain cell with too few positives as anecdotal
rather than printing a precision or a recall that a single example would
dominate.

## 3. Features

### 3.1 Where the probe reads

The probe reads the residual stream at one configured layer, at every
completion token, and maps that vector to a single logit through an optional
normalization step and one linear layer. Prompt tokens are given no target and
never contribute to a loss or a metric; only completion tokens are scored.

Activations are produced by replaying a stored rollout through the model with
teacher forcing: the prompt is rebuilt through the identical chat-template path
used at generation time, the stored completion token ids are appended
unchanged, and the whole sequence goes through one forward pass. The completion
never gets re-encoded from text, because encode-then-decode is not guaranteed
to round-trip and a silent one-token shift would misalign every activation
against its label.

The probe can optionally read a short context window of consecutive hidden
states rather than a single one, concatenated before the linear layer. The
default is a single position, so the probe's decision at token $t$ depends on
the model's state at $t$ alone.

A run may also name several layers rather than one, in which case it trains an
independent head per layer over a single pass of the data. This is worth doing
because of how the stored activations are laid out rather than for any
statistical reason: every layer of a rollout lives in one file, and opening that
file costs far more than reading it, so asking for all of the layers costs about
a fifth more than asking for one. Depth stops being an axis that multiplies the
number of runs and becomes one that can be swept inside a run.

The heads share no parameters and their losses are added. A sum of independent
terms gives each head exactly the gradient it would have received on its own,
so this is a rearrangement of training the depths separately rather than an
approximation of it, and each head is written out as an ordinary single-layer
probe. Two details are what make that equivalence hold rather than nearly hold,
and both are covered where they arise: the gradient norm is clipped per head
(Section 6.2), and each head keeps its own normalization, since activation scale
varies with depth.

Depth turns out to matter more than any other axis studied here, which is the
practical argument for sweeping it inside every run rather than fixing it once.
Between the best depth and the worst, token-level coverage at a fixed
false-alarm budget differs by more than an order of magnitude, while the knobs
of Sections 4 and 5 move it by a few points. The best depths sit around a third
of the way up the stack, the region is broad rather than a single winner, and
the last few layers are the worst by a wide margin. A late layer also converges
more slowly, so a comparison made at a fixed step can mistake a depth that has
not finished for a depth that cannot do the job.

Because a run carries every depth, a result is read at a named depth and the
name is part of the result. Reading one depth as though it settled the question
for the others is the mistake this arrangement exists to make avoidable.

### 3.2 Frozen features versus an adapted model

Two regimes exist, and which one is in use is a configuration field:

- **Frozen** (`cached` in the configuration). The language model is fixed, so
  the hidden state at every token of every rollout is a constant. Those hidden
  states are extracted once for the whole corpus and stored, and training then
  reads vectors from disk instead of running the model. Training a linear head
  on precomputed vectors is bounded by input/output rather than by compute,
  which makes large sweeps over recipes, window sizes and seeds cheap enough to
  run exhaustively.
- **Adapted.** Low-rank adapters are attached to the layers up to and including
  the probed layer and are trained jointly with the probe head. The hidden
  states now change at every optimizer step, so the cached activations cannot
  be used and each step pays a full forward and backward pass of the language
  model.

One consequence is worth stating plainly, because it is easy to get wrong when
reasoning about cost. In the frozen regime, selecting a subset of a rollout's
tokens for training genuinely reduces work, since only the selected vectors are
read. In the adapted regime it does not: attention is causal, so producing the
hidden state at a token requires running the whole prefix before it, and
selecting a subset only changes which positions contribute to the loss. Token
selection strategies are therefore a statement about *what the probe learns
from*, and only incidentally about cost.

The comparison of recipes runs in the frozen regime, where every combination
can be run at several seeds. The adapted regime is reserved for a small final
comparison, since its cost per configuration is orders of magnitude higher and
it answers a different question (does adapting the representation help) than the
recipe comparison does (which selection rule and which target help).

The two regimes also differ in how many depths a run can carry. Sweeping depth
inside a run works because the stored activations hold every layer, and an
adapted run has no stored activations, so it carries one probe at one depth and
places its adapters up to that depth. An adapted comparison therefore runs at a
depth already chosen in the frozen regime, and its control is a frozen run at
that same depth rather than one depth of a many-headed run: a many-headed run
selects its checkpoint on whichever depth leads and fits one class weight for
all of them, so it differs from an adapted run in more than the regime.

## 4. Targets

### 4.1 The label contract

Every labeling scheme produces the same two things for a rollout: a per-token
target value, and a per-token mask marking positions with no defined target.
Masked positions contribute to nothing, neither loss nor metric. This uniform
shape is what lets the token selection logic stay completely unaware of which
labeling scheme is active: selection decides *which* tokens enter a batch,
labeling decides *what value* each of them is trained toward, and neither needs
to know about the other.

Three families of labels are supported.

### 4.2 Frontier labels, hard

$$y_N(t) = 1 \iff f_r - t \leq N$$

for a horizon $N \geq 0$, with every token of a negative rollout labeled $0$.

At $N = 0$ this is the plain statement "degenerate from the frontier onward":
tokens before $f_r$ are negative, tokens from $f_r$ to the end of the rollout
are positive. Larger $N$ shifts the positive region earlier, asking the probe to
fire $N$ tokens before degeneration is confirmed. The formula needs no special
case past the frontier, since the distance $f_r - t$ simply goes negative there
and the condition stays satisfied for every horizon.

Whether the horizon is a genuine axis or only a relabeling is a question the
lead-time view settles, since a probe asked to fire earlier should be seen to
fire earlier while rollout-level detection stays roughly unchanged.

Measured that way, the horizon does nothing. Across an eightfold range of $N$,
at a window wide enough to express every horizon in it and at both anchors, the
median alarm position moves by a few tokens while coverage falls slightly. The
labels do change as intended, and the positive rate of the training stream rises
monotonically with $N$, so this is a property of the probe and not of the
labelling. The reading is that the probe fires when the loop becomes visible and
cannot be relabelled into committing earlier.

Two consequences follow. The horizon is held at zero rather than treated as an
axis to sweep, and the interesting question moves from *when the target says to
fire* to *how much of the run-up is readable at all*, which is what the frozen
against adapted comparison of Section 3.2 asks. Note that this is a statement
about the horizon, not about the run-up: the run-up does carry signal, and
Section 5.2 records how much of it different selection rules recover.

### 4.3 Frontier labels, soft

The same frontier, with the step function replaced by a decay. Tokens from
$f_r$ onward are labeled $1$; before the frontier the target decays with
distance, either linearly to zero over a configured length or exponentially
with a configured rate. Negative rollouts are labeled $0$ throughout.

The motivation is that the hard label claims a token one position before the
frontier is as innocent as one a thousand positions before, which is not what
anyone believes. The decay expresses "closer to the frontier is more
degenerate" without committing to a hard cut. The decay length is the knob that
says how far back the run-up is considered to reach.

### 4.4 Token-level signals

The target can also come from a per-token quantity computed independently of
the frontier, such as a repetition score over a sliding window or the model's
own predictive entropy. These are stored for every rollout at every token, so
they plug into the same contract directly.

They differ from the frontier families in an important way: they define a
target everywhere, including on negative rollouts, where a nonzero repetition
score describes text that is genuinely repetitive but legitimate. Training
toward them therefore teaches a somewhat different concept from "this rollout
has broken", which is exactly why the choice of target is an experimental axis
and not a foregone conclusion.

### 4.5 Losses, and the rule about correcting imbalance twice

The loss is configured independently of the label. Binary cross entropy is the
natural pairing for the hard frontier labels and also accepts the soft frontier
targets directly, since a target anywhere in $[0, 1]$ is well defined for it.
Regression losses (squared error, absolute error, smoothed absolute error)
pair with the continuous token-level signals. Keeping label and loss as two
fields rather than one bundled name is what makes combinations such as "frontier
distance trained as a regression" expressible without inventing a new name for
every pair.

Class weighting follows one rule: **imbalance is corrected once**. The positive
weight applied inside the loss is computed from the population the probe
actually sees after token selection, not from the raw corpus. Rebalancing the
sampled population and then applying a corpus-derived weight on top of it
corrects the same skew twice and pushes the probe toward firing indiscriminately.
Every run logs the realized positive rate of its own training stream alongside
the weight in force, so the product of the two stays visible and close to one.
"The population the probe actually sees" is meant literally: where a selection
rule masks the loss instead of shrinking the batch, the weight and the reported
composition are both counted over the tokens that reach the loss, not over the
pool they were drawn from.

One consequence reaches into monitoring. Because the weight is fitted to each
recipe's own training stream, and every selection rule changes that stream's
balance, two recipes score the same monitoring split on two different scales:
their weighted losses are not comparable, and a curve of weighted loss across
recipes is largely a picture of the weight. An unweighted loss is therefore
reported alongside it, measured identically for every recipe. Selection is
unaffected either way, since it runs on a rank metric (Section 6.3), but the
loss is the natural convergence signal and it should mean the same thing
everywhere it is plotted.

## 5. Training examples and batches

### 5.1 The unit is a window

A training example is a contiguous **window** of $W$ tokens from one rollout,
not a whole rollout. Every selection strategy, including the one that trains on
everything, emits the same descriptor: a rollout plus a start and end position.
This gives one code path, one batch shape and a fixed cost per example, and it
keeps gradient statistics stable. With whole rollouts as the unit, a batch
holds one or two examples of wildly different lengths and its gradient is
dominated by whichever long rollout happened to land in it.

Rollouts shorter than $W$ contribute every token they have, without padding and
without exclusion. Falling short of $W$ is a property of the rollout, not a
reason to distort or drop it.

### 5.2 The selection ladder

Five selection strategies are available. They are ordered so that each one
changes exactly one decision relative to the previous one, which is what makes
the difference between two adjacent strategies attributable to a single cause.
Every one of them draws from the same pool: the **entire** rollout, including
the region past the frontier, so that only the selection rule differs between
rungs and never the pool itself.

1. **`all_tokens`.** Every token of every rollout, tiled into consecutive
   windows, with no subsampling anywhere. Class balance is handled purely
   through the loss weight. This is the reference point: the whole population
   exactly as generated, with no assumption about which examples are useful. It
   is also the most expensive and the slowest to converge, since most tokens of
   most rollouts say nothing about the frontier.

2. **`rollout_balanced`.** Negative rollouts are subsampled to a fixed multiple
   of the positive count, and each surviving rollout contributes a fixed budget
   of $W$ tokens regardless of its true length, drawn uniformly at random over
   its whole length. Fixing the budget here means that from this rung onward
   the token count per rollout is constant and can never again be a confound.
   The question this rung isolates is whether correcting the rollout-level
   class ratio helps on its own.

3. **`random_window`.** The same budget of $W$ tokens per rollout, but now they
   must be contiguous: one window placed uniformly at random anywhere in the
   rollout. The question isolated is whether contiguity matters, holding both
   the budget and the randomness of placement fixed.

4. **`frontier_window`.** The same contiguous window, but for positive rollouts
   its position is anchored on the frontier rather than random. $W$ is chosen
   at least as large as the longest horizon of interest, so both classes stay
   representable inside a single window. Negative rollouts keep random
   placement, since they have no frontier to anchor to. The question isolated
   is whether looking specifically near the frontier beats looking anywhere.

   The anchor has two styles, itself a separate comparison. A **trailing**
   window ends at the frontier and therefore contains only run-up tokens, which
   matches the deployment situation exactly: nothing after "now" is available to
   a live decision. A **centered** window spans both sides of the frontier and
   mixes run-up tokens with confirmed in-pattern ones, which may stabilize the
   positive class at the cost of spending part of a fixed budget on the easy,
   already-degenerate region.

   The anchor and the label horizon are not independent. A trailing window holds
   only tokens before the frontier, so under a horizon of zero it contains no
   positive token at all and the recipe has nothing to learn from. That
   combination is refused when the training set is built, rather than trained on
   and reported, and the refusal names the three ways out: raise the horizon so
   the run-up is marked, use soft frontier labels, or centre the window.

5. **`frontier_window_hard_negative`.** Identical for positive rollouts. For
   negative rollouts the window is no longer placed uniformly: placement is
   biased toward spans that look structurally repetitive under the repetition
   and longest-repeated-substring signals, yet belong to a rollout that ended
   naturally. Those spans are exactly the confusable cases, such as genuine
   incremental work or repetition the prompt asked for. A configurable fraction
   of negative windows comes from this biased pool and the rest stay uniform,
   so coverage and calibration are not lost. The question isolated is whether
   deliberate exposure to repetitive-but-legitimate text reduces false alarms on
   the text those heuristics are known to misfire on.

What the ladder separates is not what it was built to separate, and the
difference changes which view it should be read in. Rollout-level detection
cannot see the ladder at all: every rung catches nearly every degenerate
rollout, so recall is saturated and the rungs are indistinguishable in it. They
split cleanly in lead time and coverage instead, and into two groups rather than
five. The three rules that sample broadly behave alike, the two anchored on the
frontier behave alike, and the gap between the groups is many times the spread
across seeds.

The anchored rules fire substantially earlier and cover substantially less. The
mechanism is not the one that trade-off first suggests. Aligning every
degenerate rollout on its own frontier and averaging shows the same monotone
rise for every rung, with no anchored rule producing a sharper peak at onset.
What differs is how far the score of a degenerate rollout sits above that of a
healthy one at the same distance before the frontier, where the anchored rules
hold roughly half again the separation. The advantage is confined to the run-up
and has gone shortly after the frontier. That is the whole of the effect: the
anchored rules read the approach better, and no rule reads the loop itself
better.

Two controls make that safe to state. A healthy rollout has no frontier to align
on, so it borrows one drawn from the frontier distribution of the degenerate
rollouts. Without that null the comparison is unreadable, because a probe's
score drifts upward with position on its own and much of the apparent rise
before a frontier is drift rather than anticipation. And the ordering holds at
every false-alarm budget and at matched coverage, so it is a property of the
probe rather than of where a threshold happened to land.

### 5.3 Choosing the window size

$W$ is not fixed a priori. A small set of candidate sizes is piloted and the
winner is then locked for the ladder. Running the full ladder at every candidate
size would multiply its cost to answer a question (how much context a window
needs) that is orthogonal to what the ladder measures (which selection rule
helps).

Two constraints bound the choice before any measurement.

$W$ has to be large enough for the window to express the horizon, or a
comparison of horizons measures the window instead. A centered window spends
half its length after the frontier and so shows only $W/2$ tokens of run-up,
needing $W \geq 2N$; a trailing window is all run-up and needs $W \geq N$. Below
that, two different horizons label every token in the window positive, train on
identical data, and are reported as two points that differ in nothing. The same
bound applies to a soft label, with the decay length in place of the horizon.

And the pilot has to run at a depth worth locking against. A window size chosen
at a depth that turns out to be poor is a window size chosen on a probe that
barely works, and there is no reason to expect it to transfer. Since a run
carries every depth at once (Section 3.1), the pilot is read at the depth the
ladder will be read at, and both are named in the result.

The two rules that place windows deliberately are the informative ones to pilot
on, since window size is the same lever those rules pull: an anchored window
trades coverage for lead time, and its size moves the probe along that same
trade-off. Where a rule ignores position entirely, $W$ only sets how contiguous
its tokens are.

### 5.4 Batch composition

Batches are assembled by an explicit composition rule, not by shuffling a flat
list of windows. Shuffling gives batches that are almost entirely negative, so
many steps carry no positive gradient at all and the per-step loss swings
violently under a positive weight. The rules are:

- A fixed ratio of positive to negative windows in every batch.
- Within the negative quota, domains are drawn in proportion to their share of
  the negative pool rather than uniformly over all negatives pooled together.
  This matters most for the hard-negative rung, because the confusable cases
  are themselves domain specific: brute-force enumeration concentrates in the
  mathematical and code domains, instructed repetition in the
  instruction-following one. Domain-proportional drawing and hard-negative
  biasing are one combined selection rule there, not two filters applied in
  sequence.
- A cap on how many rollouts of the same prompt may be used per epoch. Without
  it, a prompt whose rollouts all degenerate floods training with near
  duplicate windows.
- Window placements are redrawn every epoch rather than materialized once. This
  is free augmentation, and it makes the comparison between random and anchored
  placement a comparison over the pool rather than over one arbitrary draw.

### 5.5 Seeds

One seed per run drives probe initialization, adapter initialization, batch
order and every sampling decision. Each configuration is run at a small fixed
set of seeds and every reported metric is a mean and a standard deviation over
those repeats, so that the difference between two adjacent rungs is read against
its own noise floor instead of against a single point estimate. This deliberately
merges optimization noise and sampling noise into one number; separating them
would need per-source seed plumbing and would answer a question nobody has asked
yet.

## 6. The training loop

### 6.1 An equal budget for every recipe

The training budget is expressed in optimizer steps and tokens per step, and
both are held identical across every recipe in a comparison. This is what makes
the comparison mean anything. An epoch is not a fixed amount of learning here:
under the exhaustive strategy an epoch is the entire corpus, while under an
anchored-window strategy it is one window per rollout. Training each recipe for
"one epoch" would hand them budgets that differ by an order of magnitude, and
the measured difference would be mostly a difference in training length.

The tokens a step sees are **measured from the training stream, not inferred
from the configuration**. Gradient accumulation is sized from the mean number of
tokens each example actually contributes to the loss, which differs sharply
between rules: a rule emitting windows contributes a window, while a rule
keeping whole rollouts contributes a whole rollout, and while the model adapts a
selection rule masks the loss rather than shrinking the batch. Deriving the
accumulation from the configured window size instead would quietly hand one rule
several times the budget of another, inside the one setting held equal to make
them comparable. Every run records the realized tokens per step beside the
requested figure.

Holding both quantities fixed has a consequence worth stating, because it is the
first thing a reader asks. Since the rules produce very different amounts of
data, a rule with a large pool never finishes a pass over it while a rule with a
small pool goes round several times. What the large-pool rule trains on is then
decided by the order the data is composed in: positives are shuffled across the
whole split and negatives drawn per batch in proportion to each domain's share,
and only afterwards does the step limit cut the sequence short. Because the
shuffle precedes the cut, a run that gets through a tenth of its pool draws that
tenth from everywhere rather than from the first rollouts in file order. A rule
that completes several passes redraws its windows at each pass boundary, so it
sees freshly placed windows rather than the same ones again; comparing rules on
windows seen *per rollout* is therefore the fair reading, not on totals.

### 6.2 Optimization

Parameters are split into two groups with independent learning rates: the probe
head (its linear layer and its normalization) and, when the adapted regime is
in use, the low-rank adapters. Everything else in the language model is frozen,
and the trainer refuses to start if any parameter outside those two groups turns
out to require gradients, so a silent full fine-tune is impossible. Gradient
accumulation is used to reach the configured tokens per step.

Gradients are clipped by norm. With a head per layer the norm is taken **per
head** rather than across all of them together, and the clip is applied to the
accumulated gradient a step is about to be taken on rather than to any single
micro-batch's. A norm spanning every head would rescale all of them whenever any
one exceeded the limit, which would couple heads that otherwise share nothing
and is the single place a joint run would stop matching separate ones.

Which layers the adapters cover is its own axis: none (a strictly frozen model),
every layer up to the probed one, or an explicit list.

### 6.3 Monitoring, selection and stopping

Validation during training runs on a fixed subset of `val`, evaluated every $N$
steps. The subset is derived from the split alone and not from the run seed, so
every run in a comparison is monitored on identical rows; letting it move with
the seed would add a difference in what was measured to the difference the seeds
exist to measure. Every positive rollout is kept and only the negatives are
thinned, since positives are the scarce class and the rank metric is built from
them. Its numbers exist to steer the run and are never reported.

How far the monitor is thinned matters more than it first appears, because
training and evaluation read the corpus differently. Training reads one short
window of a rollout; evaluation reads the whole rollout. A probe covering many
depths therefore pays for every token of every layer each time it is monitored,
and full-split monitoring at a frequent cadence can cost more than the training
it is meant to observe. A convergence curve does not need the whole split. The
final numbers do, and those are measured once, at the end, on the full split.

For the same reason the end-of-run evaluation names which splits it covers. An
exploratory run has no use for the test splits, which are several times larger
than the monitor, and not reading them is the discipline Section 7.7 asks for
anyway.

The metric used to stop and to select a checkpoint is computed in **evaluation
space**, from the probe's scores through the protocol of Section 7, and never
from the training loss. The reason is that losses are not comparable across
recipes: cross entropy against hard frontier labels and squared error against a
decayed target are different currencies, and stopping each recipe when its own
loss plateaus stops each one at a different point of the same trade-off curve.
Every recipe, whatever it was trained against, produces a per-token score in
$[0, 1]$, so all of them can be judged on the same score-space quantity.

The metric is recall at a fixed false-alarm budget: the share of degenerate
rollouts caught while no more than one percent of healthy ones are allowed to
fire. It ties the checkpoint to the operating point the probe would actually run
at, and it keeps moving over the range where checkpoints differ.

A threshold-free ranking metric is the obvious alternative and is the wrong
choice here, for a reason worth recording because it is invisible until
measured. Separating a mostly-degenerate rollout from a healthy one is easy, so
the area under the ROC curve reaches its ceiling long before the probe is
useful. Across depths spanning a threefold difference in token coverage it
agrees to within a thousandth, which makes it unable to tell two checkpoints
apart exactly where the choice matters. Ranking is still worth recording, as a
health check and as the calibration-free number that survives a score-scale
shift across domains, but not as the quantity anything is selected on.

The price of an operating-point metric is that it is not invariant to a monotone
rescaling of the score, so a probe trained on a decayed target whose outputs
live in a compressed range is not automatically comparable to one trained on
hard labels. This is why the threshold is re-derived per run rather than shared,
which restores the comparison at the cost of one number per run.

The metric is a configuration field and may be something else (median lead time
at a fixed budget, token-level coverage at a fixed budget) when a different
question is being asked. The rule is that it may differ between comparison
tables and never within one.

A run carrying a head at every depth reports each metric per depth and again
without one, the latter holding the best depth at that step. Selection reads the
aggregate, since the best depth is the probe the run would be used for. Any
comparison across recipes reads a named depth, because the aggregate can move
between depths from one step to the next.

Stopping works as follows: the best checkpoint by the selection metric is kept,
training halts after a configured number of evaluations without improvement, and
there is a hard cap on steps. Early stopping chooses *which step's checkpoint is
kept* and never shortens one recipe's budget relative to another's. Each run
records the step it selected, and if runs routinely stop at the cap then the
budget is too small and the comparison is not yet valid.

Two guards run alongside:

- **Collapse guard.** A probe can minimize its loss by ignoring its input and
  emitting one constant value for every token, which under a strong positive
  weight is a real attractor. Such a probe has a respectable loss and is
  worthless, and rank-based metrics computed on constant scores are undefined
  or arbitrary. Every validation pass therefore records the standard deviation
  of the probe's scores over the monitored tokens, and a run whose score spread
  falls to approximately zero is flagged and treated as invalid rather than
  entering a results table with a plausible-looking loss.
- **Positive-rate check.** The realized positive rate of the training stream and
  the positive weight in force are logged together, which surfaces a double
  correction of class imbalance immediately.

### 6.4 What a run writes

Every run writes: the fully resolved configuration, the composition of each
split as the probe actually saw it, the selected checkpoint (the probe head,
plus the adapter weights when adapters are in use), the last checkpoint, the
training and monitoring history, and, once training is over, the decision
thresholds derived from the full validation split as described below.
Checkpoints are small, since only the head and the adapters are ever saved and
never the base model, which is what makes it affordable to save on the
validation cadence and therefore to resume an interrupted run rather than
repeat it.

A run is named after its own axes, with a fingerprint of the full configuration
attached, so two runs that differ in any setting that changes what they learn
can never be confused for one another. Repeats of one configuration share a
parent and differ by a timestamp, so running the same thing twice adds an
attempt instead of overwriting the first, and seed repeats of one recipe carry a
shared group label that aggregates them into a single line with a spread.

The metric history is mirrored locally in the same shape it is sent to the
experiment tracker, and written from the same hook, so the two cannot drift
apart. Every plot the tracker shows during a sweep can therefore be rebuilt
afterwards from the run directory alone, along with comparisons across runs that
the tracker was never asked for.

## 7. The evaluation protocol

### 7.1 Scores as the interface

Evaluation never receives a model. It receives a file of **per-token scores**:
one row per rollout, carrying the rollout's identity, its domain, its split, how
it stopped, its length, its frontier if it has one, and the sequence of
probabilities the scorer assigned to its tokens.

Everything else follows from that. The protocol is blind to how a score was
produced, so it applies unchanged to any probe recipe and to non-learned
baselines such as a repetition heuristic mapped into $[0, 1]$, which puts probes
and baselines in the same table by construction. Producing scores costs a pass
over the data on a GPU; computing metrics from them is a small job on a CPU, so
the protocol can be extended or re-run whenever a question changes without ever
recomputing a score. And because scores are stored rather than recomputed, two
metrics reported for the same run are guaranteed to describe the same numbers.

Scoring is exhaustive. Every token of every rollout in an evaluation split is
scored, with no subsampling of negative rollouts and no cap on tokens per
rollout. A cap can only underestimate a false alarm rate, and the false alarm
rate is the number the deployment decision turns on. The evaluation splits are
never resampled or rebalanced either, because a rebalanced split's rates would
no longer estimate what happens on real, naturally imbalanced generations. Only
the training split is ever resampled.

The correctness of the protocol is checked against synthetic scorers with known
answers: a perfect oracle that steps from zero to one exactly at the frontier,
an oracle delayed by a fixed number of tokens, an oracle that fires early, a
constant scorer, pure noise, and a noisy oracle that spikes for single tokens on
negative rollouts. Each has a hand-computable value for every view, which is
what makes a surprising number on a real probe trustworthy as a result rather
than suspect as an off-by-one.

### 7.2 The first-alarm position

One quantity underlies every view. For a rollout $r$, a decision threshold
$\tau$ and a persistence window $m \geq 1$:

$$
a_r(\tau, m) = \min\{\, t : p_r(t') \geq \tau \text{ for all } t' \in [t, t+m) \,\},
\quad \text{or } \infty \text{ if no such } t \text{ exists}
$$

where $p_r(t)$ is the score at token $t$. This is the first token at which the
probe would have fired, requiring $m$ consecutive tokens at or above the
threshold before it commits. At $m = 1$ this is plain first crossing. Larger $m$
trades a little lead time for immunity to a single noisy spike, which is exactly
what produces a spurious early stop on an otherwise healthy generation. Both
$\tau$ and $m$ are properties of the evaluation and not of the probe, and both
are chosen on validation data as described in Section 7.7.

The first alarm plays the same role for probes that first threshold crossing
plays for a repetition score and first match position plays for a
longest-repeated-substring signal, so probe results slot into the same table
shape as those baselines.

### 7.3 View A: rollout-level detection

At a fixed threshold, over all rollouts of a split: a rollout counts as
predicted positive when its first alarm is finite. That gives a confusion matrix
against the true rollout label, and from it precision, recall and accuracy.
Sweeping the threshold gives a rollout-level ROC curve, a precision-recall curve
and their areas, reported alongside the fixed operating point as a
threshold-free summary and as a check that no single operating point was
cherry-picked.

### 7.4 View B: token-level coverage

Rates are reported per population, with population sizes always shown next to
them, and never blended into one accuracy number.

- **Negative rollouts.** Every token is scored. The metrics are the token-level
  false positive rate and the fraction of rollouts with at least one false
  positive anywhere.
- **Positive rollouts, from the frontier onward.** Recall is what matters here:
  once inside the loop the probe should catch essentially every token of it.
  Both the pooled recall and, per rollout, the *count* of missed in-pattern
  tokens are reported, because one missed token deep inside a long loop is a
  very different failure from missing the last fifty.
- **Positive rollouts, before the frontier.** Not scored against a positive
  label, since these tokens are not degenerate yet. This population is what
  View C is about.

The reason this view is split rather than pooled is that the in-pattern tails
dominate the positive token population and are trivially separable. A single
blended token accuracy would read close to perfect for almost any probe and
would say nothing about the part of the problem that is hard.

### 7.5 View C: lead time

- For detected positives, the distribution of $a_r(\tau, m) - f_r$: negative
  values mean the probe fired before the frontier and bought lead time, positive
  values mean it fired late. Median and mean are reported, signed and absolute,
  in the same form as the onset offsets used for the heuristic baselines.
- Positives that were never detected are reported separately as such, never
  folded into an average offset, where they would silently distort it.
- For negatives, any finite first alarm is a false early stop. Its rate is
  reported directly, since it is the concrete cost of deploying the probe as an
  early-stopping trigger.

### 7.6 View D: alarm persistence

Detection and lead time say nothing about whether the probe stays convinced. A
probe that fires once and immediately retracts is noise, even when its first
alarm happens to land in the right place; a probe that fires and then holds is
making a decision. For a fixed threshold, per rollout and from the first alarm
onward:

- the length of the first firing run,
- the duty cycle after the first alarm (the fraction of subsequent tokens at or
  above the threshold), which is more robust than the run length because one
  dip should not erase the story,
- the number of separate firing episodes across the rollout, where one episode
  is the clean "decides once" behaviour,
- the retraction rate: how often the probe fires and then falls silent for the
  whole remainder,
- the gap between the first alarm and permanent commitment, meaning the first
  position after which the score never drops below the threshold again. This is
  the width of the region in which the probe is dithering.

The compact summary is a two-state view of the alarm: the probability that a
firing token is followed by another firing token, and the probability that a
quiet token is followed by a firing one. Mean run length is one over one minus
the first of those. Two numbers per threshold per population make statements
like "once firing, it keeps firing with probability 0.99, and while quiet it
fires spuriously once in ten thousand tokens" directly readable.

Three rules keep this view honest:

- It is read in opposite directions for the two populations. On positives, a
  high duty cycle and a sticky alarm are good. On negatives every alarm is an
  error, and the run-length distribution separates a jittery probe (spikes of a
  few tokens) from a confidently wrong one (long sustained runs). Those two
  failures call for completely different fixes.
- It is never reported alone. A probe that outputs one everywhere has perfect
  persistence and is useless, so persistence is always reported in the same row
  as the false alarm rate and the lead time at the same threshold.
- Run lengths are reported both raw and normalized by the number of tokens
  remaining after the first alarm, otherwise a probe that fires late looks
  artificially incoherent.

This view also decides $m$, and how it does so needs stating carefully, because
the obvious reading of the trade-off is wrong.

The tempting argument is that false alarms on negative rollouts are runs of a
few tokens while true alarms on positives run for hundreds, so a persistence
window should remove spurious early stops at a cost of a few tokens of lead
time. What that argument misses is that $m$ and the threshold are not
independent. The budget is an equation rather than an observation: the threshold
is solved for so that a fixed share of healthy rollouts fires. Requiring $m$
consecutive tokens makes firing strictly harder at any fixed threshold, so
holding the budget forces the threshold down until the same share of healthy
rollouts again reaches $m$ in a row.

Both halves of that are large and they oppose each other. Lowering the threshold
alone moves the alarm hundreds of tokens earlier and spends many times the
budget; requiring the run alone moves it hundreds of tokens later and leaves
most of the budget unspent. The net is the small difference between two large
opposing terms, which is why the curve of lead time against $m$ is nearly flat
and why its noise grows with $m$: at a persistence of one the spread across
seeds is a few tokens, and at sixty-four it is tens of tokens, larger than the
effect being measured.

So $m$ is chosen on validation alongside the thresholds, as a small window near
the low end rather than by seeking an optimum. The optimum is not identifiable
at the seed counts these comparisons run at, and reporting one would be
reporting noise. What the view is genuinely for is the diagnosis it was built
for: telling a jittery probe from a confidently wrong one, and confirming that a
first alarm is a decision rather than a blip.

Two quantities are reported with any choice of $m$, because the budget does not
constrain them. The token-level false-positive rate moves several-fold across
the range of $m$ while the rollout-level budget sits still, and the run length
of a false alarm grows with $m$, so a fixed budget at a large $m$ means a
sustained wrong alarm rather than a flicker. A budget held at the rollout level
says nothing about either.

### 7.7 What is tuned where

The boundary between tuning and reporting is enforced by the tooling rather than
by discipline.

- **Validation decides everything.** The checkpoint, the persistence window, the
  thresholds, the window size, and which recipe wins are all chosen on
  validation data. Choosing among several recipes and several seeds by their
  test numbers would be threshold shopping against a small positive population.
- **Thresholds are frozen before test data is touched.** Rather than committing
  to one threshold, a small family of operating points is fixed by targeting
  several false alarm budgets on the validation split's negative rollouts, for
  example the thresholds that hit one, five and ten percent. A budget framing
  matches the deployment question, which is how much false-alarm cost is
  acceptable, and that question cannot be settled here, so all views are
  reported at all three points together with the threshold-free areas.
- **Frozen values are written to a file** by the validation pass and read back
  by every test report. The reporting tool computes a threshold only from the
  validation split and refuses to produce a test report for a scorer that has no
  frozen thresholds, so the leak is structurally impossible rather than merely
  discouraged.

For held-out domains the frozen in-domain threshold is applied unchanged,
because that is the honest zero-shot number, but the threshold-free area is
reported per domain next to it. Score scales can shift across domains, and the
two numbers together separate a calibration shift (ranking still works, the
threshold no longer fits) from a representation failure (the ranking itself does
not transfer). They call for different fixes, and one number alone cannot tell
them apart.

### 7.8 Reporting rules

- Held-out domains are reported per domain and never pooled.
- Every rate is printed with its population sizes, in tokens and in rollouts.
- A budget smaller than the share of negative rollouts tied at a scorer's
  highest score cannot be spent: the threshold is forced past the top of the
  range and nothing fires anywhere, positives included. The resulting empty
  columns are indistinguishable from a scorer that simply stayed quiet, when the
  truth is the opposite, so the tie is measured and reported beside the
  threshold it made unreachable. This is not a hypothetical: a per-token signal
  that saturates, or a binary one, ties on most of the corpus.
- Any per-domain cell backed by fewer than a small minimum of positive rollouts
  is reported and marked anecdotal, never silently hidden and never quoted as an
  estimate of generalization.
- Every metric of a recipe is a mean and a standard deviation over its seed
  repeats. Three seeds is the minimum for a claim that one recipe beats another.
  A single-seed run is a pilot: it can establish that a configuration trains,
  and it can support a negative result when the same flat answer appears across
  many settings of the axis under study, but it cannot rank two recipes. Any
  table mixing the two says which rows are which.
- Numbers from the in-loop monitor are never reported, including in a table
  meant only to show progress. The monitor runs on a thinned split, so a monitor
  number and a protocol number for the same run and the same metric will differ,
  and putting them in one table invites a comparison that is not valid. Reported
  numbers come from stored scores through the views above.
- Any metric read at an operating point names the persistence window it was
  computed at, since lead time, coverage and the token-level false-positive rate
  all move with it while the budget does not.

## 8. Comparing recipes

A training run is a point in a grid of independent axes: the token selection
strategy, the label family and its parameters, the loss, and the adapter scope.
Any combination is evaluated by the identical protocol of Section 7, so a table
whose rows are combinations and whose columns are the four views is the natural
output of the question "which recipe is better, and in what respect".

Depth is the exception, and it is worth treating differently. Because a run can
carry a head per layer at little extra cost (Section 3.1), depth need not be
fixed before the other axes are studied and then hoped to transfer. Any recipe
can be run across every depth at once and the layer chosen afterwards, per
recipe, which also answers whether the best depth is the same for all of them.
Reading a single depth's result as though it settled the question for the others
is the mistake this makes avoidable.

Two conventions keep such a table meaningful. Within one table, only the axis
under study varies and everything else, including the training budget, the
monitoring subset, the selection metric and the seed set, is held fixed. And
differences between adjacent rungs of the selection ladder are reported as
deltas with their spread across seeds, since each such delta is attributable to
exactly one design decision only if the ladder was built so that exactly one
thing changed.

## 9. The configuration surface

A run is fully specified by one configuration. The fields below are the ones
that change what is learned or what is measured; the rest (paths, logging,
tracking) are incidental.

```yaml
features:
  regime: cached            # cached (stored activations) | adapted (run the model)

probe:
  layer: 12                 # which residual stream the probe reads
  layers: null              # or several: a head per depth, trained in one pass;
                            # cached only, since adapted carries one probe
  context_window_size: 1    # consecutive hidden states concatenated per decision
  normalization: layernorm  # none | layernorm | rmsnorm | l2
  dtype: float32

label:
  family: frontier_hard     # frontier_hard | frontier_soft | token_signal
  horizon: 0                # frontier_hard: y=1 iff f_r - t <= horizon
  decay: exponential        # frontier_soft: linear | exponential
  decay_length: 256         # frontier_soft: how far back the run-up reaches
  signal: repetition_score  # token_signal: repetition_score | entropy

loss:
  name: bce                 # bce | mse | l1 | smooth_l1
  bce:
    use_pos_weight: true    # weight fitted to this recipe's own stream
  mse:
    output_activation: sigmoid

selection:
  strategy: frontier_window # all_tokens | rollout_balanced | random_window |
                            # frontier_window | frontier_window_hard_negative
  window_size: 128
  anchor: centered          # frontier_window: trailing | centered
  positive_fraction: 0.25
  max_rollouts_per_prompt: null
  hard_negative_fraction: 0.5
  resample_each_epoch: true

lora:
  enabled: false            # adapted regime only
  layers: all               # none | all | [list of layer indices]
  rank: 16
  alpha: 32
  dropout: 0.05

budget:
  tokens_per_step: 2048     # measured from the training stream, not from W
  patience: null            # evaluations without improvement before stopping
  collapse_threshold: 0.01  # minimum score spread on the monitor

optimizer:
  probe_learning_rate: 1e-4
  lora_learning_rate: 1e-4
  weight_decay: 0.0

runtime:
  max_steps: 800
  per_device_train_batch_size: 8
  max_grad_norm: 1.0        # applied per head, after accumulation
  seed: 42

validation:
  strategy: steps
  steps: 200
  max_rollouts: 400         # thin the monitor, keeping every positive
  final_splits: []          # which splits the end-of-run pass covers; empty
                            # skips it, for a run scored from a checkpoint later

checkpoint:
  strategy: steps
  steps: 200                # must match the validation cadence to keep the best
  metric_for_best_model: recall_at_budget
  greater_is_better: true

dataset.sampling:
  train_negative_rollouts_per_positive: 4.0
  evaluation_negative_rollouts_per_positive: null   # evaluation never subsamples
  domain_stratified: true
```

Evaluation is not configured here. It runs over stored scores after training,
so its budgets, its persistence windows and the splits it covers are arguments
to the reporting step rather than properties of a run: false-alarm budgets of
one, five and ten percent, a persistence sweep, per-domain reporting, and a
minimum positive count below which a cell is marked anecdotal.

## 10. Reproducibility

A run is reproducible from its resolved configuration and its seed. The
configuration is stored next to the checkpoint and mirrored to the experiment
tracker, so a checkpoint always carries the exact settings that produced it.
Seeds cover initialization, batch order and every sampling decision, so two runs
of the same configuration and seed see the same windows in the same order.
Evaluation is reproducible independently of any of that, because it consumes
stored scores: a metric can be recomputed, corrected or extended long after the
GPU that produced the scores is gone, and every metric reported for a run is
guaranteed to be derived from the same numbers.
