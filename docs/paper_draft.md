# Early Warning Is Not a Property of the Probe

_Draft: abstract, introduction, related work, methods. Numbers marked
`[PENDING]` are experiments in flight or not yet run. Every other number is
measured on the validation split of the Apertus-8B-Instruct build and is
provisional until the frozen test is read once._

---

## Abstract

Language models sometimes fall into repetitive loops and never recover. Recent
work reports that these loops are visible in the model's internal states, and
in
some cases can be predicted before the repetition appears in the text. We ask
what evidence would support that stronger claim, and whether the usual
measurement can distinguish it from cheaper explanations. We build a corpus of
36,300 sampled generations in which degeneration occurs naturally in 2.3% of
answers, with loop onsets located semantically by a prompt-aware judge and
aligned to token positions, and we score every token of every answer at fixed
per-answer false-alarm budgets. Three measurement problems inflate the usual
numbers, all present in our own pipeline. Answer-level detection is saturated:
rollout recall stays between 0.94 and 1.00 across the 29 of 31 depths that
clear an eligibility floor, while coverage of the run-up varies by a factor of
25 over those same depths. The natural
per-token form of a repetition baseline reads the window ahead of the token it
labels; correcting it leaves detection unchanged, removes three quarters of its
apparent early warning, and moves its median alarm from 66 tokens before the
loop to 178 tokens after it. And the annotated onset is itself uncertain by a
margin comparable to the window over which early warning is measured. Under a
protocol that separates tokens by position relative to the onset, linear probes
on the residual stream recover the loop reliably once it has begun (in-pattern
coverage 0.82) but flag only 4.0% of the 256 tokens before it, against 3.6% for
the corrected baseline. We then decompose that residual. Matched for absolute
position, run-up tokens are 48 times more likely to be flagged than healthy
tokens between positions 500 and 1000, and no more likely at all past position
2000. Answer length explains most of the false alarms. A healthy answer is 2.1
to 3.4
times more likely to raise an alarm when another answer to the same prompt
degenerated. And one domain of five, code, supplies between 47% and 88% of
every flagged
run-up token while being 21% of the degenerate answers, ranking highest on
run-up coverage in all twelve configurations compared. We
conclude that a pre-onset signal exists but is concentrated rather than
general,
that a single global operating point is the wrong object for it, and we specify
the controls that stronger claims should be required to pass.

---

## 1. Introduction

Language models sometimes stop making progress and begin repeating themselves.
Once this starts they rarely recover, and the generation runs until it hits the
token limit. The user waits longer, pays more, and gets nothing usable. The
failure is old and well documented, and it has not gone away in current models.

A natural response is to catch it from the inside. Hidden states are computed
anyway during generation, so a small classifier reading them costs almost
nothing, and several recent papers show that this works. Reported accuracies
and
areas under the curve for separating looping from normal states sit above 0.99.
Some of this work goes further and reports that a loop can be predicted before
the repetition is visible in the text.

The second claim is the one worth having. Detecting a loop that has already
started saves the remaining tokens. Predicting one before it starts would let a
serving system act while the answer is still good. It is also the harder claim
to establish, and we argue it is currently under-tested. Our starting point is
a
single question: if a probe appears to warn early, what else could explain
that,
besides the model representing an impending loop?

We find four alternatives, and each predicts the same headline numbers.

**The task is nearly saturated.** In our corpus a degenerate answer spends
about
four fifths of its tokens inside the loop, because it is by construction one
that ran to the token cap and the loop typically begins about a fifth of the
way
in. An answer scored by its highest-scoring token is therefore easy to
classify.
Our probes reach rollout recall between 0.94 and 1.00 at a 1% false-alarm
budget, and so do configurations we consider poor. Over those same depths, the
share of the run-up that gets flagged moves by a factor of 25. A metric that is
flat where the interesting quantity moves that much cannot rank methods. The
fix
is not a better metric but a finer one: we score tokens separately depending on
whether they fall before or after the annotated onset.

**The obvious baseline reads the future.** Standard repetition metrics are
defined over a completed sequence. Comparing one against a probe needs a value
at every token, and the natural implementation computes the score over the
window ahead of the current position. That window contains the loop, so the
baseline appears to anticipate a failure it is already observing. Replacing the
forward window with a trailing one leaves detection identical, at 52 of 108
degenerate answers caught either way, removes about three quarters of the
apparent early warning (0.149 to 0.036 coverage of the 256 tokens before the
onset), and moves the median alarm from 66 tokens before the loop to 178 tokens
after it. Every comparison in this paper uses the corrected version.

**The label is uncertain by roughly the width of the claim.** Our onsets are
marked by a judge that reads prompt and answer and quotes the text where the
loop begins. Compared against independent structural signals, the disagreement
has a median absolute value of tens to hundreds of tokens, while the window
over
which we measure early warning is 128 to 256 tokens. We treat this as a
quantity
to propagate rather than assume away, and report how far our conclusions move
when onsets are shifted. `[PENDING: human validation of a stratified sample of
onsets, with inter-annotator agreement, and a sensitivity analysis over that
range.]`

**The probe may be reading the situation rather than the trajectory.** Every
degenerate answer in our corpus runs to the token cap, while healthy answers
are
shorter and stop on their own, with a median length of 474 tokens and fewer
than
1% reaching 2000. A probe that learned "this answer is long and still going"
would reproduce our headline numbers without encoding anything about
degeneration. This concern is not hypothetical in the probing literature, where
classifiers at 100% accuracy have been shown to fall to chance once response
length and source identity are removed. Because we sample ten answers per
prompt, a second version of the same concern is directly testable: whether the
probe fires on healthy answers whose prompt siblings degenerated.

Closing these four gaps is what this paper does, and doing so turns a single
number into a decomposition. We find that a pre-onset signal exists, and that
it
is concentrated rather than general.

**Contributions.**

1. A corpus of 36,300 sampled generations at a naturalistic degeneration rate of
   2.3%, with verified, semantically located loop onsets, ten answers per
prompt,
   and seven source domains of which two are held out entirely.
2. An evaluation protocol for pre-onset detection: positional decomposition
   around the onset, per-answer false-alarm budgets frozen on validation, and a
   causally valid repetition baseline. The uncorrected baseline overstates
early
   warning by a factor of about four.
3. A decomposition of the apparent pre-onset signal into absolute position,
   answer length, prompt-level predisposition and domain, with the residual
   measured rather than asserted, including where that residual is zero.
4. A systematic sweep over which tokens the probe learns from and what it is
   trained toward, at matched compute, showing which design choices move the
   pre-onset number and which do not.
5. `[PENDING: cross-model result. Probes trained directly on Llama-3.1-8B-Instruct
   and Mistral-7B-Instruct-v0.1 builds, to separate "the direction does not
   transfer" from "these models do not degenerate readably".]`

We are explicit about what this does not show. We do not show that pre-onset
detection is impossible. We show that the evidence usually offered for it does
not separate it from cheaper explanations, that when those are controlled what
remains is small and unevenly distributed, and that the places where it
survives
are specific enough to name.

---

## 2. Related Work

**Text degeneration.** Repetitive degeneration is a long-standing failure of
neural text generation, usually measured after the fact. The standard metrics
count duplicate n-grams in a finished text and produce one number per completed
sequence. This is useful for comparing decoding strategies and unusable during
generation, since the metric needs the whole text before it can be computed.
Turning such a metric into a per-token signal is a design choice rather than a
standard, and Section 3 shows that the obvious version of that choice looks
ahead at tokens the model has not yet produced. `[VERIFY: Holtzman et al. 2020;
Welleck et al. 2020; Li et al. 2023; Fu et al. 2021. These come from an earlier
literature review and have not been re-checked.]`

**Detecting a loop from internal states.** Several recent papers show a loop
can
be read off hidden states once it has started. Duan et al. (2026) train linear,
SVM and MLP classifiers on hidden states and report accuracy and AUC above 0.99
for both loop types they study. Xie et al. (2025) train a linear classifier on
the hidden state of the token ending each reasoning chunk and report about 92%
accuracy at temperature zero. `[VERIFY: this figure and Duan et al.'s 0.99 come
from a secondary summary and need checking against the papers.]` Yu et al.
(2025) detect recurrent generation from
activation patterns at 95.24% accuracy, in a setting where loops are induced
adversarially.

We take these results as settled and do not try to improve on them. Our own
detection numbers agree with them. The point is different: this task is close
to
saturated because of the shape of the data, so a high answer-level number says
little about whether the model gives any warning before the loop begins.

**Predicting a loop before it starts.** The closest work is Duan et al. (2026),
who predict loop onset from hidden states by combining their classifier with a
cumulative-sum change-point rule, and report early detection rates and lead
times
in tokens. Our results do not contradict theirs. They are measured under
different conditions, and the conditions are what this paper is about. Their
detector operates at a reported false-positive rate well above what a serving
system would accept, on a balanced test set drawn from a benchmark constructed
to
trigger loops, using greedy decoding and sentence-level representations. We fix
a
budget of 1% of healthy answers, preserve a natural 2.3% base rate, sample at
temperature 0.7, and score every token. These are different points on a
trade-off, not competing measurements of one quantity, and we present them that
way. `[VERIFY: their exact early detection rates, false-positive rates and lead
times, from Section 4 of the paper rather than from a summary.]`

Duan et al. also report that semantic circularity appears before verbatim
repetition. We do not dispute that a precursor exists. We ask the narrower and
more testable question of whether that precursor is linearly decodable from the
residual stream, per token, at a false-alarm rate a deployment would accept.

**Detection strength grows inside the loop.** Xie et al. (2025) frame their
finding as models being self-aware when trapped in repetition. Their own
appendix
supports a more careful reading: tracking one example, their classifier scores
the first repetitive chunks near zero and reaches certainty only hundreds of
chunks later. This is detection strength increasing with depth into the loop,
and
it is the distinction we build the protocol around. Being self-aware once
trapped and being self-aware on the approach are two claims, and only the first
is currently supported.

**What probes measure.** Our protocol follows a line of work on how to read
probe
results. Hewitt and Liang (2019) show that a probe can reach high accuracy on
random labels, so accuracy alone does not establish that a representation
contains the target. Sahoo et al. (2026) give a recent and sharp example:
linear
probes reaching 100% accuracy on a reasoning-type task fall to chance once
source
identity and response length are residualised out. This is directly relevant,
since every degenerate answer in our corpus runs to the token cap while healthy
answers are shorter. Fomin et al. (2026) make the closest methodological
argument, asking when an internal readout supports a claim about what happens
next rather than describing the current situation, and report negative results
across three probe families. Their distinction is the one we draw between
in-pattern coverage and warning coverage. Our contribution is not that framing,
which is theirs, but its instantiation in a setting where the onset can be
located, which lets us measure the gap rather than only argue for it.

**Probe robustness.** Kramár et al. (2026) report that probes trained on
short-context data fail under production distribution shift, especially on long
contexts. Our position finding gives a concrete mechanism for that observation
in this task: the share of tokens flagged in a healthy answer rises by more
than
two orders of magnitude between the start of an answer and position 2000.

**Mechanisms behind repetition.** A separate line of work asks why repetition
happens rather than whether it can be seen, localising it to specific neurons
or
sparse features, or tracing it to repetition in the training data. These are
complementary. We do not propose a mechanism. We measure how much of the
phenomenon is visible to a linear readout before it appears in the text, and
under what measurement conditions that visibility survives. `[VERIFY: Hiraoka
and
Inui 2025; Yao et al. 2025.]`

**Judge-produced labels.** Using language models as annotators is now common,
and
the accepted practice is to validate them against human labels with a
chance-corrected statistic and to report human agreement alongside. Most of
that
literature validates a verdict. Ours produces a position, which then becomes
both
a supervision target and the reference point for every metric we report, so
onset
error propagates differently and we treat it accordingly.

---

## 3. Methods

### 3.1 Corpus

We generate 36,300 answers from Apertus-8B-Instruct-2509 by sampling ten
continuations for each of 3,630 prompts at temperature 0.7 and top-p 0.9, with
a
hard cap of 4,096 tokens and a fixed seed. Prompts are drawn from seven sources
spanning mathematics, competitive programming, general instruction following
and
medical reasoning. Five are used in-domain and split by prompt into training,
validation and an in-domain test set. Two, competitive programming problems
rated 2000 and above and a medical reasoning set, are held out entirely and
read
only for zero-shot evaluation.

Sampling rather than greedy decoding is a deliberate choice with two
consequences we rely on. It matches the generation policy a served model would
use, and it produces several trajectories per prompt, which is what makes
prompt-level effects measurable at all.

### 3.2 Labels and the degeneration frontier

An answer that ended naturally at an end-of-sequence token is a **negative**.
An
answer that was cut off by the token cap is sent to a judge, which reads the
prompt and the answer and decides whether the text stopped making progress. The
prompt is essential: a list, a code template, a brute-force enumeration or
repetition the prompt explicitly asked for are all legitimately repetitive, and
a purely structural rule cannot tell them from a loop.

When the judge finds degeneration it returns a short verbatim quote marking
where the pattern first begins. Asking for a quote rather than a number is what
makes the label checkable. A separate step locates the quote in the answer's
own
token stream and records the index of its first token, giving the **frontier**
for that answer. An answer whose quote cannot be located is excluded from every
split rather than given an invented onset.

Of the 36,300 answers, 890 hit the cap. Of those, 818 yield a usable frontier
and become positives, a rate of 2.3%. The remainder are excluded: 63 where the
judge returned nothing usable, 8 judged not degenerate, and 1 whose quote could
not be found. Splits are assigned at the prompt level, so all ten answers to a
prompt share a split.

The single most important fact about the corpus is where loops start. Across
the
positives the quartiles of the onset position are 399, 721 and 1,144 tokens,
and
about 3% begin at the very first token. Since every positive runs to 4,096
tokens, roughly four fifths of a positive answer is already loop. This is what
makes answer-level classification easy and answer-level metrics uninformative.

### 3.3 Probe

A probe is a learned normalisation followed by one linear map, reading the
residual stream at a single layer at a single token, and emitting a
probability.
It has about 12,000 parameters per depth. The score at a token depends on that
token's state alone, which is what makes it deployable one token at a time, and
the state is built only from tokens up to that position, so a probe never reads
text the model has not yet produced.

Activations are extracted once with the model frozen and stored, so training
reads vectors from disk. This makes it affordable to train an independent head
at every layer from 1 to 31 in one pass, and to repeat every comparison at
three
seeds. Depth then becomes an axis inside a run rather than one that multiplies
the number of runs, which matters because depth turns out to move the results
more than any other choice we study.

Training holds two things fixed across every recipe so that comparisons mean
something: the number of optimizer steps and the number of target tokens per
step, measured from the training stream rather than inferred from the
configuration. Batches are composed to a fixed positive fraction rather than
shuffled, since shuffling a population that is over 99% negative leaves most
steps with no positive gradient. Evaluation is never rebalanced, because a
false-alarm rate measured on a rebalanced population is not the rate a
deployment sees.

### 3.4 Evaluation protocol

Evaluation never receives a model. It receives one score per token, so probes
and non-learned baselines go through an identical judgement by construction.
Scoring is exhaustive: every token of every answer in a split, with no
subsampling of negatives and no cap, since a cap can only understate a
false-alarm rate.

Rather than choosing a threshold directly, we fix the share of healthy answers
allowed to raise a false alarm, at 1%, 5% and 10%, and read off the threshold
that spends exactly that. Thresholds are chosen on validation and reused
unchanged everywhere else. The validation population is 3,634 answers, 108 of
them degenerate.

Four quantities are reported at each operating point.

- **Rollout detection**: whether the probe fires anywhere on a degenerate
  answer. This exists to confirm a scorer is not broken, not to rank scorers.
- **In-pattern coverage**: the share of tokens at or after the frontier that are
  flagged. The easy half, and a floor to clear rather than a result.
- **Warning coverage**: the share of tokens in a band of 128 or 256 immediately
  before the frontier that are flagged. This is the quantity the paper is
about.
- **Lead time**: the signed distance from the first alarm to the frontier, with
  the share of answers that fired before the frontier and the count that never
  fired reported beside it, since the offset distribution is bimodal and a
median
  over it alone is misleading.

The healthy token flag rate is reported alongside all of these, since flagging
every token maximises both coverages.

### 3.5 Controls

The decomposition is what separates a claim about the trajectory from claims
about the situation, and each control removes one alternative.

**Position.** A probe's score climbs with position on its own. We therefore
compare run-up tokens against healthy tokens at the same absolute position,
grouped into bands, so that the two populations in a comparison are read at the
same depth into a generation.

**Length.** Every positive runs to the cap while healthy answers are much
shorter, so healthy answers are grouped by their own length before their
false-alarm rates are compared. The share of each domain reaching the lengths
at
which false alarms occur is reported alongside, since a high rate in a band a
domain rarely reaches contributes few alarms.

**Prompt predisposition.** Because ten answers share a prompt, healthy answers
whose prompt also produced a degenerate answer can be compared against healthy
answers from prompts that never looped. A probe reading imminence should score
them the same, since nothing went wrong in either. The two populations are
compared by the probability that a random member of the first outscores a
random
member of the second.

**Domain.** Every quantity is reported per domain with its population sizes,
and
any cell backed by fewer than ten degenerate answers is marked anecdotal rather
than quoted as a rate.

**Permuted onsets.** `[PENDING: a control in which frontiers are permuted among
the degenerate answers, preserving the onset distribution so the positional
statistics of the control match the real condition. This separates a signal
about degeneration from what supervised training extracts from any sufficiently
flexible representation, and is the direct analogue of a control task.]`

### 3.6 Checkpoint and depth selection

A run trains for 2,000 steps and keeps a checkpoint every 50. Which one a depth
is judged on is decided by a rule applied to each depth independently: a depth
becomes eligible once its in-pattern coverage clears a floor, and among
eligible
steps the one with the best warning coverage is kept.

Selecting on answer-level recall instead is a mistake we can quantify, because
that metric saturates within the first few evaluations. On the configurations
we
report, selecting by recall lands on checkpoints between steps 50 and 500, and
those checkpoints have 6.8 times lower warning coverage, lower in-pattern
coverage, and a first alarm 81 tokens later than the ones the rule selects.
Depth is chosen the same way, on validation, and reported as part of the
result.

### 3.7 Adapted representations

`[PENDING: retrained low-rank adapter probes. The existing runs show large
gains,
roughly 0.06 to 0.22 in warning coverage on the leading configuration, but they
are not matched to their frozen controls on batch composition, target tokens
per
step, or checkpoint cadence, and only three checkpoints per run were scored.
The
question the reworked runs answer is not whether adaptation helps but whether
it
improves the residual that survives the controls of Section 3.5. The one signal
available so far points the other way: the prompt-predisposition effect is not
smaller under adaptation.]`

---

## 4. Results

`[PENDING: to be written against the frozen test split, read once, after
selection is closed and recorded. The validation figures quoted in the abstract
and introduction are provisional and will be replaced. If the test weakens any
claim, the claim changes rather than the framing.]`

Planned structure:

1. Baselines and the cost of the lookahead correction.
2. Detection saturates while warning coverage varies by depth.
3. What the design sweep moves and what it does not.
4. The decomposition: position, length, prompt, domain.
5. The out-of-sample test of the domain result on the held-out code domain.
6. Adaptation, read through the same controls.
7. Cross-model generalisation.
