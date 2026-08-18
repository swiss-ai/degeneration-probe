# What this project is arguing, and why it is new

A note on the idea behind the paper, for someone who knows the project but has
not followed every experiment.

---

## The short version

A language model that falls into a repetitive loop rarely recovers. Recent work
shows that a small classifier reading the model's hidden states can spot such a
loop, and some of it goes further and claims the loop can be seen coming before
the repetition appears in the text. That second claim is the valuable one,
because a monitor that only fires once the answer is visibly broken saves the
remaining tokens and nothing else.

Our question is what evidence would support the second claim, and whether the
usual way of measuring it can tell the difference. The answer is that a
pre-onset signal does exist, but it is far smaller and far more concentrated
than a single headline number suggests, and most of what looks like early
warning is explained by things that have nothing to do with an imminent loop.

The sentence we can defend is this:

> Early warning is not a property of the probe. It is a property of the domain,
> the position in the answer, and the prompt.

---

## Why the usual measurement cannot settle it

Three problems make the standard numbers look far better than they are, and all
three are visible in our own data.

**The task is nearly saturated.** A degenerate answer runs to the token cap,
and
its loop typically starts about a fifth of the way in, so roughly four fifths
of
it is already loop. Score an answer by its highest-scoring token and almost any
scorer catches almost every degenerate answer. Across every depth that clears
the eligibility floor, 29 of 31, our rollout-level recall stays between 0.94
and 1.00, while the share of the
run-up that gets flagged varies by a factor of twenty-five over the same
depths.
A number that is flat where the interesting quantity moves by that much cannot
be used to compare anything.

**The obvious baseline reads the future.** A repetition metric is defined over
a
finished text. Turning it into a per-token signal is a design decision, and the
natural implementation computes the score over the window ahead of the current
token. That window contains the loop, so the baseline appears to anticipate a
failure it is already reading. Moving the same window behind the token leaves
detection untouched and removes about three quarters of its apparent early
warning, while its median alarm moves from 66 tokens before the loop to 178
tokens after it.

**The reference point is uncertain.** Our onsets come from a judge that reads
prompt and answer and quotes where the loop begins. Compared against the
structural signals, the disagreement is tens to hundreds of tokens, which is
the
same order as the window over which early warning is measured. This is a
quantity to propagate rather than to assume away.

---

## The questions only our setup can ask

This is the part that makes the work a contribution rather than a critique.
Several plausible explanations for an apparent early warning are testable only
if the corpus was built a particular way, and ours was.

| how the corpus is built | what it makes answerable |
|---|---|
| ten sampled answers per prompt | whether the probe reads the prompt's own tendency to loop, rather than the trajectory in front of it, by comparing healthy answers whose siblings degenerated against healthy answers from prompts that never did |
| natural rate of degeneration, several sources | whether early warning is a general ability or something that happens in one kind of text, and what a false alarm rate means on a realistic population |
| an onset located semantically, by a judge reading the prompt | whether the signal is real or positional, because the reference point is not defined by the same surface statistic being measured |
| a score for every token | matched comparisons at all, since a token can be placed both relative to the onset and relative to the start of the answer |

Benchmarks assembled to trigger loops have no natural base rate and no domain
mixture. Greedy decoding produces one trajectory per prompt, so the sibling
comparison does not exist. Balanced evaluation sets remove the false-alarm
question. An onset defined by a repeated n-gram or by chunk similarity makes
the
label and the baseline the same object.

So the framing is not that earlier work is wrong. It is that these questions
cannot be asked inside those designs, and here are the answers.

---

## What the decomposition finds

Taking the apparent pre-onset signal apart along each axis in turn:

**Position accounts for most of it, and all of it late in an answer.** The
share
of tokens flagged in a healthy answer climbs steadily with position, from
effectively zero in the first few hundred tokens to about one in eight past
token two thousand. Comparing run-up tokens against healthy tokens at the same
absolute position, the run-up is roughly fifty times more likely to be flagged
between tokens 500 and 1000, six times between 1000 and 1500, three times
between 1500 and 2000, and no more likely at all past 2000. There is a real
residual, and it disappears exactly where the probe is busiest.

**Length accounts for the false alarms.** Every degenerate answer in the corpus
runs to the cap while a healthy one stops on its own, with a median of about
470
tokens and fewer than one percent reaching two thousand. On healthy answers
alone, the correlation between length and score is around 0.67, and the longest
quarter of them absorbs almost the entire false-alarm budget.

**The prompt accounts for a further part.** A healthy answer is 2.1 to 3.4
times more likely to raise an alarm if another answer to the same prompt
degenerated, across the three budgets tested, and the effect is overwhelming
statistically. Letting a low-rank adapter reshape the representation does not
reduce this and slightly increases it.

**And one domain carries most of what remains.** Code is a fifth of the
degenerate answers but supplies between 47% and 88% of every flagged run-up
token, and it ranks highest on run-up coverage in every one of the twelve
candidate configurations we compared. It is also the lowest-scoring domain
inside the loop in nine of those twelve, which no single notion of a domain
being easy explains. The effect survives both controls: at matched
position code sits between about fifty and several hundred times above its own
healthy null, where the mathematics domains sit at a factor of a few and
instruction-following inverts outright, and at matched length code produces the
fewest false alarms of
any domain despite having by far the most long healthy answers.

The practical consequence is concrete. A single global threshold is the wrong
object, because the same detector is plausibly useful on code and useless on
mathematics, and the budget it spends is spent almost entirely on long answers
in the domains that carry no signal.

---

## What we are claiming

1. A corpus and an evaluation protocol built so that "does the probe see the
   loop coming" can be decomposed into questions that have answers.
2. The decomposition itself: how much of the apparent pre-onset signal is
   position, length, prompt tendency and domain, and how much is left.
3. Two measurement failures that inflate results in this area, both found in our
   own pipeline and both general: a per-token repetition baseline that reads
   ahead of the token it labels, and rollout-level metrics so saturated that
they
   cannot pick a checkpoint. Selecting on rollout recall lands six to seven
times
   worse on run-up coverage than selecting on the quantity we care about.

## What we are not claiming

We are not claiming that pre-onset detection is impossible, that our detector
is
better than anyone else's, that we have found a mechanism, or that any of this
generalises beyond the model family we measured. We are also not claiming the
protocol as a result. It is the method.

---

## Where we sit

Two bodies of work matter here and they have not met.

The first is loop detection from internal states, which reports accuracies and
areas under the curve above 0.99, and in one case predicts loop onset well
before the text repeats. The second is work on what probes actually measure:
control tasks, probes that reach perfect accuracy and fall to chance once
response length and source identity are removed, probes that describe the
current situation rather than predicting the next event, and probes that fail
under a shift in context length.

The second literature predicts precisely what we found in the first one's task,
and nobody has connected them. That is the gap. Our closest neighbour reports a
genuine precursor, and we do not dispute it; we ask the narrower and more
testable question of whether that precursor is linearly readable per token, at
a
false-alarm rate a serving system would accept, on a population with a
realistic
base rate. Their operating point and ours are different regimes rather than
competing measurements, and the paper should say so plainly.

---

## What is still open

- Whether the concentration is about code or about one source. A held-out code
  domain that never entered training is the out-of-sample test, and it is the
  single result that decides which paper this is.
- Whether a probe trained directly on other model families behaves the same way,
  which is what the ongoing rebuilds are for.
- Whether adaptation improves the residual we care about or only reads the
  confounds better. The evidence so far leans towards the second.
- A control in which the onsets are permuted among answers, which separates a
  signal about degeneration from what supervised training extracts from any
  flexible representation.
- Human validation of the onset, and the frozen test, read once.
