# LRS and the LLM judge

This note explains two of the metrics we use to detect degeneration in model
completions: **LRS** (Longest Repeated Substring), an algorithmic metric, and
the **LLM judge**, a model-based metric used to check the algorithmic ones
against something closer to ground truth. Numbers below come from all three
dataset builds (`apertus-8b-instruct`, `apertus1p5-capfilter-linear-it8816`,
`apertus1p5-sft256k-4200`), computed in
`degeneration_probe/dataset_gen/label.py`, `llm_judge.py`, and
`notebooks/inspect_dataset.ipynb` (Sections 6–7), and given as a range across
the three builds where they differ.

[ADD THE CONTEXT THAT THE INFORMATION FROM THIS NOTE WILL BE USED AS CONTEXT FOR THE LATEX DOCUMENT FOR THE FINAL PAPER OF THIS PROJECT
THE LATEX DOC IS STILL DON'T HAVE AN INTRODUCTION AND THE ONLY THING EXPLAINED RIGHT NOW IS TTR.
ALL METRICS THAT WE ARE USING TO ANNOTATE THE DATASET (ENTROPY, TTR, LRS, LLM-JUDGE).
HERE IS THE STORY THAT I WOULD LIKE TO BE TOLD: 
- BY ANALYZING JUST THE DEFINTION OF DEGENERATION TWO FIRST IDEAS COME TO MIND DEPENDING ON THE POINT OF VIEW:
    - LOOKING AT WHAT HAPPENS INSIDE: DEGENERATION IS BASICALLY A COLLAPSE IN THE TOKEN'S PROBABILITY DISTRIBUTION ONTO A SINGLE/BUNCH OF TOKENS => LOOK IF ENTROPY CORRELATES WITH DEGENERATION. IT DOES, BUT FULL OF FALSE POSITIVES AND NOT REALLY RELIABLE
    - LOOKING AT WHAT IS GENERATED: A DEGENERATE TEXT IS REPETITIVE, HERE WE COULD LOOK FOR THE EXACT REPEATED PATTERN (LRS) OR SIMPLY LOOK AT PLACES IN TEXT WHERE THE REPETITION OF TOKENS IS HIGH (TTR).
BY MANUAL INSPECTING A FEW GENERATIONS AND LABELS, WE NOTICED THAT EACH METRIC WAS NOT PERFECT: FULL OF FALSE POSITIVES AND COULD NOT REALLY DETECT THE EXACT BEGINNING OF THE DEGENERATION. WE DECIDED TO USE LLM-JUDGE TO VALIDATE THE METRICS. BY DOING THIS ANALYSIS WE CONLCUDED THAT NO METRIC CAN BE BLINDLY TRUSTED AND THUS WE DECIDED TO KEEP USING LLM-JUDGE AS FINAL METRIC (AND THEN ALSO MANUAL INSPECTED). THIS IS EXPENSIVE BUT STILL DOABLE AS IN OUR CURRENT DATASET ONLY 1000 SEQUENCES WERE POSITIVE.


THIS IS THE STORY I WOULD LIKE TO TELL, SO WHEN I'LL ASK TO WORK ON THE LATEX WE WILL HAVE TO CREATE ALL NECESSARY SECTIONS AND TO KEEP IN MIND THIS STORY WHILE EXPLAINING THE METRICS. TO GIVE SOME MORE CONTEXT, WE'VE SPLIT THE WORK BETWEEN TEAMMATES AND I ONLY HAVE TO WORK ON INTRODUCITON + LRS + LLM-JUDGE. THE TTR (THAT IS THE ONLY METRIC ALREADY PRESENT IN THE LATEX) AND ENOTRPY ARE NOT FOR ME TO DO. DON'T TRUST THE ALREADY PRESENT TEXT ABOUT TTR AS IT WAS DONE WITHOUT MUCH ATTENTION BY MY COLLEAGUE AND SHOULD NOT BE TAKEN AS EXAMPLE OF WHAT WE REALLY WANT. JUST FOLLOW MY STYLE. 
]

[DON'T TREAT THE FOLLOWING TEXT AS EXACTLY WHAT I WANT TO INCLUDE INSIDE THE LATEX DOCUMENT, BUT JUST A CONTEXT TO TAKE INFORMATION FROM. I THINK IN THE LATEX WE CAN REDUCE THE LEVEL OF DETAIL OF A FEW PASSAGES AS NOT FUNDAMENTAL FOR THE STORY, BUT STILL YOU HAVE ACCESS TO ALL INFO FOR CONTEXT. IN ADDITION YOU CAN STILL ALWAYS GO READ THE EXACT SCRIPT TO HAVE BETTER CONTEXT]

## 1. LRS (Longest Repeated Substring)

### 1.1 What it is, and why we added it

The other repetition metrics we use (e.g. a sliding-window type-token ratio)
give a *continuous* score at every token position: how repetitive the local
context is. They are good at telling *whether* a rollout looks repetitive,
but they don't directly tell us *which exact chunk of text* is the one being
repeated, or where it starts.

LRS was added to answer that more specific question algorithmically: scan
the whole completion and find the longest stretch of tokens that occurs
**twice, without overlapping**. The idea is simple — if a model is stuck in a
loop, the piece of text it keeps repeating is, by definition, a long
substring that appears more than once. If it isn't looping, the longest
thing that happens to repeat by chance is usually short and unremarkable (a
common phrase, a boilerplate line).

Why is it enough to search for something that repeats just **twice**, rather
than building in a notion of "repeats many times" from the start? Because a
short unit repeating many times in a row automatically produces a long
repeat under this definition too — the repeats themselves combine into a
bigger one. Say a model loops on a short unit `P` six times back to back:

```
P P P P P P
```

Split this run in half: the first three copies (`P P P`) and the last three
copies (`P P P`) are themselves identical strings, sitting one after the
other, not overlapping. So this is already exactly the thing LRS looks for —
a non-overlapping repeated substring — just a longer one, made of three
copies of `P` instead of one. Since the binary search always keeps the
*longest* valid repeat it can find, it naturally reports this big
three-copies-vs-three-copies match rather than a single copy of `P`, with no
special-casing needed for "the same short thing repeated many times." Once
that big match is found, a second step (Section 1.5) works out its smallest
internal repeating unit and counts how many times it recurs, so the result
is reported in the more useful form "unit `P`, repeated 6 times" rather than
just "a chunk of length `3·|P|` that repeats once."

The problem is that a degenerate loop is rarely a *perfectly* identical
chunk copied twice. Very often, something small changes between one
repetition and the next — most commonly a number. For example, a model
enumerating cases might produce:

```
N = 41: checking divisibility... N = 42: checking divisibility... N = 43: ...
```

Here the model is really looping on the same template, but an exact-match
algorithm sees `"N = 41: checking divisibility..."` and `"N = 42: checking
divisibility..."` as two *different* strings, because one digit differs. The
longest exact repeat it can find is therefore only the part of the template
that is identical on both sides, which understates how much of the rollout
is actually degenerate — or, if the changing number sits in the middle of
the template, can break the match entirely.

### 1.2 A partial fix: normalizing numbers

One fix is to replace each number with a generic placeholder token before
comparing, so `"N = 41"` and `"N = 42"` look identical to the matcher.

Even with this fix, a real gap remains: numbers are not the only thing that
changes between repetitions. The model can also vary a letter, a variable
name, a unit of whitespace, punctuation, or word choice from one loop
iteration to the next — and because LRS only ever looks for **exact**
matches, any such difference is enough to break the match, or to shrink the
longest exact repeat it finds down to whatever fragment happens to be
identical on both sides. This is a harder problem than numbers, because
there's no fixed, enumerable set of "things that vary" the way there is for
digits — it can be anything.

### 1.3 How well it performs

**Detecting real degeneration.** We only send the LLM judge the rollouts
that hit the generation's hard token cap (4096 tokens) without reaching a
natural end-of-sequence token — this "truncated" population is where real
degeneration loops overwhelmingly show up. On that population, a
completion-level confusion matrix — the structural heuristic (defined
precisely in §2.5) versus the judge's verdict, computed per domain and
totaled in `notebooks/inspect_dataset.ipynb` (Section 7) — shows **precision
between 97% and 100%** and **recall between 85% and 96%** across the three
dataset builds: when the heuristic flags a rollout as degenerate it is
almost always right, but it also misses a modest share (4–15%) of the
rollouts the judge itself calls degenerate. So on the population where
degeneration is actually expected, LRS (as part of that combined heuristic)
is a good, if imperfect, detector.

**False positives, and why they happen.** The problem is elsewhere: LRS is
*guaranteed* to find some repeated substring in almost any long enough piece
of text, because with a low minimum length (currently 10 tokens) some short
phrase or formatting pattern will coincidentally repeat somewhere just by
chance — that's simply how a "find the longest exact repeat" algorithm
behaves, independent of whether the rollout is actually degenerating. We can
see this directly: among rollouts that ended normally (reached
end-of-sequence on their own, i.e. almost certainly *not* degenerate), LRS
still reports a match of 10+ tokens in **65–74% of cases**, with a typical
matched length of only ~19–21 tokens out of up to 4096 — an incidental
match, not a real loop.

A simple fix for this is to only trust LRS's signal on rollouts that hit the
4096-token cap. Among those, the picture is completely different: LRS finds
a match in **~99.2–99.9%** of cases, and the matched span's typical length
jumps to **~1,400–1,700 tokens** — roughly a third or more of the whole
completion. In other words, restricting to capped rollouts turns "LRS found
*a* repeat" from a nearly-meaningless fact into a strong, high-confidence
signal, at the cost of not scoring rollouts that ended naturally at all.

**Onset position.** Even on rollouts LRS correctly flags, it doesn't always
agree with the LLM judge on exactly *where* the degeneration begins. Because
LRS only reports positions where an exact (or digit-normalized) match is
first found, and the judge is free to point at the conceptual start of the
pattern, the two onset positions differ, on the matched-and-onset-defined
population and totaled across domains, by a mean of **~175–380 tokens**
(median ~29–92 tokens) even in the digit-normalized variant, which is
currently the best of the LRS-based onset signals we have. This absolute and
signed error, computed per domain and totaled, is saved as reusable code in
`notebooks/inspect_dataset.ipynb` (Section 7) alongside the confusion matrix
above, so both update automatically as the dataset grows.

### 1.4 Why we don't just add more tolerance

A natural next step would be to let the matcher tolerate a handful of
different tokens between two occurrences, not just digits. We chose not to
do this, for a few reasons:

- It makes the algorithm significantly more complex and computationally
  heavier — an exact match can use hashing and a clean binary search over
  candidate lengths (see below); an approximate match generally can't.
- It immediately raises a question that has no principled answer: *how many*
  differing tokens should be allowed before two spans stop counting as "the
  same repeated chunk"? Too strict and we're back to the current problem;
  too loose and almost anything starts looking like a repeat.
- That tolerance would become a tunable parameter with no natural, correct
  value — it would have to be picked empirically and would likely need
  re-tuning per model/domain. That makes LRS less of a clean, principled
  metric and more of an ad hoc heuristic, which is why we kept it as an
  exact-match algorithm (with the narrow, well-motivated exception of digit
  normalization) rather than a general fuzzy-match one.

### 1.5 Algorithm (pseudocode)

The core routine finds the longest length `L` for which some pair of
non-overlapping, exactly matching windows of length `L` exists, using binary
search over `L` (the existence of a repeat of length `L` is monotonic: any
valid repeat of length `L` contains a valid repeat of every shorter length
at the same two positions) together with a rolling hash so each probe is
linear in the sequence length:

```
function find_longest_repeated_substring(tokens, min_length):
    n = length(tokens)
    if n < 2 * min_length:
        return NO_MATCH

    precompute rolling-hash prefix sums over tokens      # O(n), done once

    function exists_non_overlapping_repeat(L):
        buckets = empty hash map                          # hash -> list of start positions
        for i in 0 .. n - L:
            h = hash_of_window(tokens, i, L)               # O(1) via prefix sums
            for p in buckets[h]:
                if i - p >= L and tokens[p : p+L] == tokens[i : i+L]:
                    return (p, i)          # verified match: hash hit + direct comparison
            append i to buckets[h]
        return NOT_FOUND

    lo, hi = min_length, n // 2
    best = NO_MATCH
    while lo <= hi:
        mid = (lo + hi) // 2
        pair = exists_non_overlapping_repeat(mid)
        if pair is not NOT_FOUND:
            best = (length = mid, first_start = pair.p, second_start = pair.i)
            lo = mid + 1        # try to find an even longer repeat
        else:
            hi = mid - 1        # no repeat this long; try shorter
    return best
```

The digit-normalized variant runs the same routine on a transformed copy of
the sequence, then maps the result back to real token positions:

```
function digit_run_collapsed(tokens):
    collapsed = empty list
    position_map = empty list          # collapsed index -> (real start, real length)
    for each maximal run of consecutive digit tokens in tokens:
        append WILDCARD to collapsed
        record (run's real start, run's real length) in position_map
    for every other (non-digit) token:
        append it unchanged to collapsed
        record (its real position, 1) in position_map
    return collapsed, position_map

function find_longest_repeated_substring_normalized(tokens, min_length):
    collapsed, position_map = digit_run_collapsed(tokens)
    match = find_longest_repeated_substring(collapsed, min_length)
    return map(match, back to real token positions using position_map)
```

Both variants also decompose the matched span into its true repeating unit
(its own smallest internal period) and re-scan the whole sequence for every
occurrence of that unit, so a chunk repeated many times back-to-back is
reported as one span with a repeat count, not just the first two occurrences
found.

## 2. LLM judge

### 2.1 Model and setup

We use **Claude Sonnet 5** as the judge, called either directly through the
Anthropic API with structured JSON output, or through the Claude Code CLI
(so calls draw on subscription usage instead of metered billing) — both
paths use the same system prompt, user prompt, and output schema, so results
are directly comparable regardless of which one produced them.

### 2.2 System prompt

The judge is given only the prompt and the completion text — never the
heuristic scores, so its verdict is not biased by them:

```
You are judging whether a language model's completion degenerated into
repetitive, low-information text (looping on a phrase, sentence, or pattern
without making progress), as opposed to repetition that is a normal,
expected part of the task. You will see both the prompt and the completion
-- read the prompt first, since it changes what counts as degeneration.

Two things that look repetitive but are NOT degeneration:

1. Genuine incremental work. Brute-force search, case enumeration, or a
step-by-step derivation often produces a fixed template with a changing
value each time (e.g. "N = 41: ..." then "N = 42: ..." then "N = 43: ...").
This is fine as long as each occurrence is doing new work -- checking a new
candidate, advancing to the next step, covering a new case. A useful test:
if you deleted every occurrence but the first, would the completion lose
information needed to solve the task? If yes, it's legitimate enumeration,
not degeneration. Only flag it if the "progress" is illusory -- e.g. it
starts re-checking values it already covered, drifts into a template that
stops advancing, or cycles without ever converging on an answer.

2. Instructed repetition. If the prompt itself asks for repeated output
(e.g. "repeat X ten times", "write a line for each of the following ..."),
producing that repetition is correct instruction-following, not
degeneration -- unless the completion overruns what was asked (keeps going
well past the requested count) or the repetition itself becomes corrupted
or loops in a way the prompt didn't ask for.

Completions are all hard-capped at the same fixed max-token budget. A
completion that ends abruptly mid-sentence, mid-word, or mid-token is
expected simply because it hit that cap -- this is not, by itself, evidence
of degeneration one way or the other. Judge based on the quality and content
of the text alone, not on whether or how it was cut off.
```

The judge is asked to return, as structured JSON:
- `reasoning` — written *before* the verdict, so it works as genuine scratch
  space rather than a post-hoc justification;
- `is_degenerating` — the boolean verdict;
- `onset_quote` — if degenerating, a short verbatim quote (~5–15 words)
  marking where the pattern *first* begins, not where it recurs;
- `confidence` — high / medium / low, lower when e.g. the token cap cuts the
  rollout off before the pattern's fate is clear.

Before writing `onset_quote` into that final answer, the judge (when run
through the `claude_agent_sdk` backend) must first call a `verify_onset_quote`
tool with its candidate quote. The tool does not run arbitrary code — it just
checks the candidate against the real completion text (a plain verbatim
substring search, plus a minimum-length check) and reports back FOUND / NOT
FOUND / TOO SHORT, so the judge gets one chance to fix a bad quote (shorten
it, fix a transcription slip) before finalizing its answer. This exists
because an `onset_quote` reproduced purely from the model's memory of the
text, with no such check, was only verbatim-locatable in the completion
about 69% of the time.

### 2.3 Why we trust it

We use it because it is the only metric we have that can reliably tell apart
**real** degeneration from things that merely *look* repetitive by
construction — the clearest case being brute-force problem solving, which is
correctly, legitimately repetitive (the same template, a new value checked
each time) and gets flagged as degenerate by every purely structural metric
we have (repetition score, LRS). This is exactly what point 1 of the system
prompt above is written to handle: because the judge understands the
*semantics* of the reasoning, not just the surface repetition of tokens, it
can tell a loop that keeps re-deriving the same thing apart from one that is
actually making progress case by case.

### 2.4 Cost and feasibility

We only send the judge the rollouts that hit the 4096-token generation cap
without reaching a natural stop — the population where degeneration
overwhelmingly concentrates (`stop_reason == "length"` alone was separately
validated at 99.6% precision against LLM-judge ground truth). In practice
that's a small slice of the full dataset: **890/36,300 (~2.5%)** of rollouts
for `apertus-8b-instruct`, **1,348/36,300 (~3.7%)** for
`apertus1p5-capfilter-linear-it8816`, **1,193/36,300 (~3.3%)** for
`apertus1p5-sft256k-4200`. Keeping the judged (and therefore
degenerate-candidate) population this small keeps the API/subscription cost
affordable, and — just as important — keeps the resulting set small enough
for us to manually read through and spot-check individual verdicts, which
would not be realistic if every rollout in the dataset were judged.

### 2.5 Reliability

We check the judge's reliability against two independent signals, both
computed in `notebooks/inspect_dataset.ipynb` (Section 7) so the numbers stay
current as the dataset grows:

- **Agreement with the structural heuristics.** "Structural heuristics" here
  means the same purely-structural signals from Section 1 and the sliding
  TTR score, combined into one flag: a rollout counts as heuristically
  flagged if its repetition score crosses the calibrated threshold *or* its
  LRS repeating unit recurs at least a handful of times back-to-back. On the
  population the judge actually sees (rollouts that hit the token cap), the
  judge's verdict agrees with this flag **~84–96% of the time**, varying by
  dataset build. This is the same confusion matrix referenced in §1.3 — that
  section reports it as precision/recall, this "agreement rate" is the same
  matrix's overall accuracy.
- **`onset_quote` verbatim rate.** Of the `onset_quote` values the judge
  actually returns, **~99.6–100%** are found verbatim in the completion text
  they were judging (checked independently of the judge's own
  `verify_onset_quote` tool call, by re-searching the final answer's quote
  against the raw completion).

The one reliability gap worth calling out: `onset_quote` can contain a
generic placeholder that happens to coincidentally match unrelated text
elsewhere in the completion — observed once as the literal word `"test"`,
which matched an unrelated "ratio test" phrase in a convergence problem.
The current pipeline (`degeneration_probe/dataset_gen/llm_judge.py`) guards
against this with two checks before a degenerating verdict is accepted: the
`verify_onset_quote` tool call itself rejects anything under
`MIN_ONSET_QUOTE_WORDS` (4) words, and a server-side check
(`_onset_quote_is_valid`) independently re-verifies the *final* `onset_quote`
field — not just whatever candidate the tool call happened to see — against
the real completion text, on both length and verbatim presence. A row that
fails either check is recorded as `"failed"` rather than `"ok"` and gets
automatically re-judged on the next run.

A second, unrelated failure mode shows up as a hard error rather than a bad
verdict: a small fraction of rollouts are refused outright by the
`claude_agent_sdk` backend (routed through the real `claude` CLI, subject to
its own usage-policy screening — separate from the raw Anthropic API), with
`"Claude Code is unable to respond to this request, which appears to violate
our Usage Policy"`. Across the three dataset configs this hits **194/3,431
(~5.7%)** of judged rollouts, and it is not a per-account rate-limit issue —
rotating through all four subscription OAuth accounts resolves almost none
of it. The refusals are overwhelmingly concentrated in one source domain:
**193/194 (99.5%)** come from `if_sft_data_verified`
(`allenai/IF_sft_data_verified`), even though that domain is only ~35% of
the judged sample. Only a small minority of the refused prompts or
completions contain topically sensitive words (9/194 and 17/194
respectively), so it isn't primarily about subject matter. The likely
driver is structural: this domain's prompts carry unusual formatting
constraints ("start every sentence with the word X", "respond entirely in
capital letters", "include at least N `[placeholder]` tokens"), and when the
model degenerates under one of these constraints the resulting completion
is a wall of near-identical repeated text (e.g. "Kill the confusion. Kill
the extra words. Kill the unnecessary sentences...", or a completion tiled
with `[KEY] [PASSWORD] [CONFIDENTIAL]`-style placeholders) — a shape that
plausibly reads as spam/prompt-flooding to the classifier regardless of
actual content. Rows hit by this are recorded as `"failed"` and, unlike
usage-window exhaustion, do not resolve on retry with a different account;
we currently accept them as permanently unjudged and exclude them from the
calibration set rather than chase a fix in the judging pipeline.

[IN THE LATEX DOCUMENT FIND THE BEST WAY TO TELL THIS INFORMATION, I'M NOT SURE HOW IMPORTNAT THEY ARE AND IF IT'S NECESSARY TO EXPLAIN THEM IN DETAILS, JUST KNOW THAT THEY ARE PRESENT]