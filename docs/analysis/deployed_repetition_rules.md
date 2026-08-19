# Nobody has measured the repetition rules that are already deployed

A short note on a result that sits slightly outside the paper's main argument.
It is written down here so the decision about where it goes can be made later,
and so it is not lost if the paper has no room for it.

---

## The observation

Thresholding a repetition statistic is the standard way to decide that a
generation has broken. It is used in three distinct roles, and in none of them
is the rule validated against an independent reference.

**As a label.** LoopGuard defines a loop offline as a unigram type-token ratio
below 0.2, a compression ratio below 0.12, and a length within twenty tokens of
the generation cap. Its online trigger recomputes the first two over a sliding
window of 256 tokens and votes two of three against a confidence streak. The
thresholds are set by inspection and no detection rate or lead time is reported
for the trigger.

**As a corpus filter.** The Gopher pipeline discards a document when any of
thirteen thresholded repetition statistics is exceeded, with cutoffs from 0.30
on duplicate line fraction down to 0.10 on duplicate 10-gram character
fraction. The same family survives in several later corpora. The thresholds were
never checked against a reference label because none existed at that scale.

**As a stop condition in serving.** vLLM terminates a request when a repeated
n-gram pattern recurs a configurable number of times, three in its own example.
The DRY sampler penalises in proportion to the length of the matched repeat.
Neither ships a published false-positive rate.

So a rule of this shape decides what gets trained on, what gets generated, and
when generation stops, and its error rate is unmeasured everywhere.

## What this corpus can say about it

The judged population is the one place a surface rule is least trustworthy and
most consequential: long generations that reached the token cap, where the
question is whether the repetition is a loop or legitimate work. A judge that
reads the prompt can separate those two, and the structural rules can then be
scored against it.

Three numbers follow, all on the released checkpoint unless stated.

- **Detection against the judge.** Balanced accuracy 0.78 to 0.91 for the
  windowed repetition score across builds, 0.88 to 0.95 for the longest repeated
  substring, and 0.63 to 0.69 for inverted entropy. Ranked consistently in that
  order in every domain, not only in the pooled view.
- **Firing on naturally terminated text.** 7 to 13% for the windowed repetition
  score, 2 to 5% for the longest repeated substring, 21 to 39% for entropy. This
  is the rate at which a deployed filter would discard, or a stop condition would
  interrupt, a generation that ended cleanly on its own.
- **Firing on legitimate long work.** Restricted to capped generations the judge
  confirms are *not* degenerate, the windowed repetition score still fires on
  79.5 to 100% of them.

The third number is the one no one else could have produced, because it needs a
semantic label on exactly the population where the surface rule is weakest. It
says that the rule which decides when to stop generating fires on essentially
every legitimately repetitive long answer.

## What it does not say

The comparison bounds the *joint* error of a structural rule and the judge, not
the error of either alone. Without human validation of the judge on this
population, a disagreement cannot be attributed. That is the first thing to fix
if this becomes a section rather than a note.

These numbers also come from a different protocol than the probe results: a
threshold tuned for balanced accuracy on a calibration half, not a threshold
solved for a per-answer false-alarm budget, and a rollout-level decision rather
than a token-level one. The two sets must not be placed in one table.

## Where it could go

**As an appendix**, which is the safe choice. It documents why the corpus is
built the way it is: why a naturally terminated answer is treated as a negative,
why only capped answers are judged, and why a judge rather than a structural
rule is the arbiter. That is the appendix a reader asks for on reaching the
sentence "an answer that ended naturally is a negative".

**As a short subsection**, if the paper wants a second, independent reason to
distrust surface repetition rules. It pairs naturally with the observation that
the same rules cannot spend a strict false-alarm budget at all, since most
healthy answers tie at the top of their range.

**As its own paper**, if the labeling work grows. The three roles above, one
corpus, one semantic reference, and the first measured error rates for rules
that are already in production is a self-contained contribution. It would need
the human validation above, and it would need the rules reimplemented exactly as
their sources define them rather than as this project defines them.

## What would have to be checked first

- The LoopGuard thresholds and window size quoted above are from a secondary
  reading and need confirming in the paper itself.
- The vLLM parameter names and default example count need confirming against the
  version cited.
- The rules should be reimplemented to their own published definitions, not
  approximated by this project's variants, before any number is attributed to
  them by name.
