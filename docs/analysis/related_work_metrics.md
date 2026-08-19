# Where our labeling metrics sit in the literature

A reference note for two decisions: whether the dataset-and-labeling results
belong in the paper, and how each of the three structural metrics should be
introduced, named, and cited. Everything below was checked against the primary
sources; the reference list at the end records what each source actually says,
so the LaTeX can cite it without re-checking.

---

## 1. Should the labeling results go into the paper?

Yes, but not as results. They belong in two places, and the split matters
because the two documents currently measure different things under similar
names.

**Into Methods, as label validation.** The paper's central claims all rest on
one object: a semantically located onset, treated as a reference point rather
than as ground truth. Two of the labeling numbers are load-bearing for that and
are currently asserted in the abstract without support:

- the judge's reliability (verbatim onset rate 99.6-100%, the server-side
  re-verification, the refusal rate and where it concentrates);
- the onset disagreement between the judge and the structural rules, which is
  what licenses the statement that the reference point is uncertain by tens to
  hundreds of tokens, the same order as the 128-256 token window over which
  early warning is measured.

The onset-agreement table is the direct evidence for the second, and it should
move into the paper more or less as it stands. One phrasing caution: a
judge-versus-metric disagreement bounds the *joint* error of the two, not the
judge's error alone. Say that, or the claim invites the obvious objection.

**Into an appendix, as corpus documentation.** The per-metric detection tables,
the ROC curves, the flag rate by termination reason and the hard-negative table
are all about why the population is constructed the way it is: end-of-sequence
as a safe negative, token cap as the gate for judging, and a judge rather than a
structural rule as the arbiter. That is exactly the appendix a reviewer asks for
when the paper says "an answer that ended naturally is a negative". The
three-build comparison also lives here, since the paper itself reports one
build.

**What does not transfer.** The detection numbers are not a result of the paper.
The paper's own argument is that answer-level detection is saturated and cannot
rank scorers, so importing a table that ranks three scorers by rollout-level
balanced accuracy works against the framing unless it is explicitly framed as
label-construction evidence rather than as a detector comparison.

### The one real conflict to reconcile

The windowed repetition score is defined over a *forward* window: position `t`
is scored on tokens `t` to `t+W-1`. The paper's second contribution is that a
per-token repetition baseline built this way reads the future and inflates early
warning by roughly a factor of four.

Both are correct, and the distinction is worth stating explicitly rather than
quietly fixing, because it sharpens the point:

- As an **offline label** on a finished rollout, a forward window is fine.
  Nothing about labeling is causal, and a forward window is the natural choice
  when the question is "does the repetition start here".
- As an **online per-token score** compared against a probe, the same formula is
  invalid, because the window it reads has not been generated yet.

So the correction is not that one implementation was wrong, it is that a metric
defined for offline annotation cannot be reused as a streaming baseline without
changing the window direction. That is a cleaner and more general finding than
"we had a bug", and it is worth one sentence in both documents.

Practical consequence: every windowed-repetition number carried over from the
labeling work must be marked as the offline variant, and must not be compared
against the paper's corrected baseline numbers, which are measured on a
different population (validation split, per-answer false-alarm budget) under a
different threshold rule (budgeted, not balanced-accuracy-tuned).

---

## 2. The repetition score is not ours

This is the main thing worth knowing: the formula is a recombination of two
established metrics, and citing them is strictly better than defending an
invention.

**The score itself is `seq-rep-2`.** Welleck et al. (2020) define

```
seq-rep-n = 1 - |unique n-grams(x)| / |n-grams(x)|
```

which is the portion of duplicate n-grams in a sequence. Our per-position score
is

```
r_t = 1 - (# distinct bigrams in window) / (W - n + 1)
```

Same formula, same denominator convention (n-gram occurrences, not tokens),
`n = 2`, evaluated on a window instead of the whole sequence. The complement is
Li et al.'s (2016) `distinct-n`, unique n-grams over total n-grams, which is the
standard diversity metric in generation work; `rep-n = 1 - distinct-n` is used
under that name throughout the repetition literature, including in recent code
work.

Calling this `1 - TTR` is the part that reads as home-made, because TTR is a
type/token ratio over words and this is a type/occurrence ratio over bigrams.
The two coincide only at `n = 1`.

**The sliding window is MATTR.** Covington and McFall (2010) introduced the
moving-average type-token ratio precisely because raw TTR is confounded by
sample length: they slide a fixed-width window one token at a time, compute TTR
inside each window, and average. Our construction is the same window, one step
per token, with two deliberate differences: the score is `1 - ` the ratio, so
that high means repetitive, and the per-window values are kept as a sequence
rather than averaged into one number, because we need a position, not a summary.
Fixed-width windowing is also what makes the score comparable across rollouts of
very different length, which is the same reason MATTR exists.

**Recommended framing for the LaTeX.** Introduce it as a local, per-position
form of the standard n-gram repetition rate, cite Li et al. (2016) and Welleck
et al. (2020) for the ratio and Covington and McFall (2010) for the moving
window, then state the two things that are genuinely ours: keeping the window
sequence instead of averaging it, and calibrating the cutoff per build against a
judge. Renaming it from TTR to something like *windowed rep-2* or *local
repetition rate* removes the last bit of apparent novelty and makes the metric
recognisable to anyone from the generation literature.

**Window size has precedent worth quoting.** MATTR windows in corpus
linguistics are typically 50-100 tokens; Welleck et al. use a 128-token window
for their token-level metric. Our `W = 256` is already at the large end of that
range, which is a useful thing to say out loud given that we are considering
larger windows: the argument for going bigger has to be about the scale of the
loops we care about, not about matching precedent, because precedent points
smaller.

**One baseline worth adding.** The literature's standard *per-token* repetition
signal is not windowed rep-n at all, it is Welleck et al.'s `rep/l`: the
fraction of predictions that occur in the previous `l` tokens, with `l = 128`.
It is backward-looking by construction. That matters twice over. It is prior art
for the corrected, trailing-window baseline, so the correction stops looking
like a repair of our own metric and starts looking like alignment with the
standard definition. And it is a cheap second baseline that no reviewer can call
arbitrary, since it is the metric the degeneration literature already uses at
token level. Implementing it as a trailing binary indicator, averaged over a
trailing window, is a few lines on top of what exists.

---

## 3. Who else uses a repetition score, and for what

The windowed repetition score turns out to be in wide use, in three distinct
roles. What nobody does is validate it, which is where our labeling work lands.

### As a label, with a hand-set threshold

- **LoopGuard (2026)** is the closest match to our construction anywhere in the
  literature, and it is concurrent. Its offline loop label is
  `Loop = 1 <=> TTR <= 0.2 and CR <= 0.12 and T >= 2480`, where TTR is the
  unigram type-token ratio over the whole generation, CR is a lossless
  compression ratio, and the length term is a near-cap gate (their cap is 2500).
  Its online trigger then computes windowed TTR and windowed CR over a
  **sliding window of W = 256 tokens** of the partial output, votes two of three
  signals, and adds as the third signal a **confidence streak**: top-1
  probability above 0.9 for more than six consecutive steps. Warmup 64 tokens,
  32-step cooldown.

  So a concurrent paper independently arrived at all three of our metric
  families: a windowed lexical-diversity score at the same window size, a
  probability-collapse signal, and a token-cap gate on the label. Three
  differences are ours: their score is unigram TTR rather than n-gram
  repetition, all their thresholds are fixed by inspection rather than
  calibrated, and their window is necessarily trailing because it runs during
  decoding.

- **Gopher / MassiveText (Rae et al., 2021)** built the ancestor of this rule at
  web scale, as a discard label rather than a degeneration label: thirteen
  thresholded repetition statistics per document, including duplicate line
  fraction 0.30, duplicate paragraph fraction 0.30, top 2-gram character
  fraction 0.20, and duplicate n-gram character fractions falling from 0.15 at
  n = 5 to 0.10 at n = 10. The same family of statistics survives in FineWeb,
  RedPajama-v2 and Dolma quality signals. Documents, not windows, and no
  reference label to check against.

- **Production data pipelines** ship the windowed version directly. Alibaba PAI's
  n-gram repetition filter defines its ratio as the cumulative frequency of
  n-grams occurring more than once over total n-gram frequency, computed with a
  sliding window at configurable n, and drops documents outside a user-set
  range.

- **Duan et al. (2026)** and **Xie et al. (2025)** label with repetition rules of
  a different shape: a repeating-unit count in the first case, chunk embedding
  similarity at 0.99 in the second. Neither uses an n-gram ratio.

- **Yu et al. (2025)** are worth singling out because they use no structural
  metric at all. Their label is the token cap plus manual review: they checked
  the Llama2-7B-chat outputs that reached `max_new_tokens` and report that all of
  them contained repeated content, differing mainly in the length of the
  repeated segment, then repeated the exercise on 800 sampled inputs across
  eight models with the same conclusion. That is independent support for our
  population design, and their qualitative note that the repeats differ "only in
  the table name" in each repetition is the same variation problem digit
  normalization addresses.

### As an evaluation metric, after the fact

This is the mainstream use, and the one our own metric descends from:
`seq-rep-n` and `rep/l` (Welleck et al.), `distinct-n` (Li et al.), the
`diversity` product over `rep-n` used in contrastive-search work, and
`rep-n`, `rep-line` and `sim-line` in the code-repetition literature. One
number per finished generation, used to compare decoding strategies or training
objectives, never used to decide whether a particular generation is degenerate.

### As an online stop condition, in serving stacks

- **vLLM** ships `repetition_detection` on `SamplingParams`: a scheduler-level
  stop condition that watches for a repeated n-gram pattern in the output tokens
  and terminates with `FINISHED_REPETITION`. It is parameterised by
  `min_pattern_size`, `max_pattern_size` and `min_count`, where `min_count` is
  the number of times the pattern must recur, for instance 3.
- **llama.cpp** ships DRY sampling, which penalises in proportion to the length
  of the matched repeated sequence.

Both are our LRS decision rule, not our windowed score: a repeated unit plus a
repeat count, with a cutoff in the same place we put ours (3 or 4 repeats).

### What is missing, and what that means for us

Nobody measures how well any of these rules actually work. LoopGuard reports no
detection rate and no lead time for its trigger. Gopher never validated its
thresholds against a reference label. vLLM's stop condition has no published
false-positive rate. The rules are deployed on the strength of looking
reasonable.

That is the gap our labeling section fills, and it is a better reason to keep
those results in the paper than corpus documentation alone. We measure, against
an independent semantic reference, a rule that concurrent work deploys with
fixed thresholds: balanced accuracy 0.78 to 0.91 for the windowed repetition
score depending on build, an end-of-sequence flag rate of 7 to 13%, and a flag
rate on judge-confirmed-negative capped rollouts of 79 to 100%. That last number
is the one nobody else could have produced, because it needs a semantic label on
exactly the population where a surface rule is least trustworthy, and it says
that the deployed rule fires on essentially every legitimately repetitive long
generation. Frame it that way and the labeling work stops being setup and starts
being a finding.

---

## 4. Entropy has precedent, but not as a label

Entropy is well grounded in this literature, though almost always as an
explanation or as a decode-time control signal, never as an annotation rule.
The chain of prior work is:

- **Holtzman et al. (2020)** show that the probability of a repeated phrase
  increases with each repetition, for the large majority of phrases tested and
  regardless of phrase length. This is the positive feedback loop, and it is the
  canonical citation for the probability-collapse mechanism our entropy section
  describes.
- **Xu et al. (2022)** name and quantify the self-reinforcement effect: the more
  times a sentence appears in the context, the higher the probability of
  generating it again. They also give the useful contrast that human text almost
  never does this (0.02% consecutive sentence-level repetition in Wikitext-103).
- **Fu et al. (2021)** give a theoretical account, the high-inflow problem, and
  characterise repetition through an average repetition probability rather than
  through entropy.
- **Li et al. (2023)** show that these accounts, high-inflow words, the
  likelihood objective and self-reinforcement, all reduce to one data-side
  explanation, and that penalising repetition in the training data is the common
  active ingredient.
- **Xu et al. (2023)** are the closest runtime use: Look-back decoding tracks
  the KL divergence between the current next-token distribution and the
  distributions of previous steps, and uses that distance to pre-empt repetition
  and topic drift. A distributional statistic monitored online, but a distance
  between distributions rather than the entropy of one.
- **Recent monitoring work** uses token-level entropy from the next-token
  distribution directly as a cheap real-time signal, though for context
  degradation rather than for loops.
- **Duan et al. (2026)** describe the loop as a *state collapse* with distinct
  boundaries, which is the internal-view framing our entropy section opens with.

Two things follow for the paper. First, the entropy section can be introduced as
the mechanism the literature already agrees on, which is a much stronger opening
than presenting entropy as one of three things we happened to try. Second, our
entropy result is a contribution rather than a footnote: we appear to be the
first to evaluate entropy as a *detector* against an independent semantic
reference, and it does poorly, with specificity around 0.53 on the released
checkpoint and an end-of-sequence flag rate of 20-39%. That is a clean
explanation of why the prior work uses entropy to *intervene* and not to
*decide*: a sustained entropy drop is ordinary in healthy text. Worth saying
plainly, since it is a negative result nobody else has measured.

---

## 5. LRS is the rule the closest prior work already uses

LRS looked like the most exotic of the three metrics. It is the opposite: it is
the exact formalisation of how the two nearest papers define a loop, and both
leave the definition informal in a way LRS fixes.

**Duan et al. (2026)** label loops in LoopBench with a repeating-unit rule. A
numerical loop requires `k * l > 500`, where `l` is the length of the minimal
repeating unit and `k` is the repetition count; a statement loop requires
`k > 3` at sentence granularity, with the threshold chosen empirically. That is
our pipeline exactly: find the repeat, reduce it to its smallest internal unit,
count how many times the unit recurs, threshold the count. The differences are
in our favour. Their `l` comes from a hand-picked granularity per loop type
(digits for one, sentences for the other), and their thresholds are set by
inspection; ours is computed exactly over token sequences at any unit length,
with the cutoff calibrated per build against a judge rather than assumed.

**Yu et al. (2025)** define recurrent generation as a subsequence
`S = [t_i, ..., t_{i+k}]` that is repeated *with slight variations* as
`S' = [t_j, ..., t_{j+k}]` with `j > i + k`, and additionally require that the
generation hit the token limit. Two things to take from this. Their definition
is a non-overlapping repeated substring, so LRS is a direct, exact
implementation of a definition already in use. And "with slight variations" is
left entirely informal there, which is precisely the gap digit normalization
fills: it makes one specific, dominant class of variation explicit and
measurable instead of leaving it to a similarity threshold. Their token-limit
requirement is also independent support for gating the judged population on the
generation cap.

**Deployment practice points the same way.** vLLM ships
`repetition_detection` as a scheduler-level stop condition: it watches for a
repeated n-gram pattern in the output and terminates the request, parameterised
by a minimum and maximum pattern size and a required repeat count, with 3 given
as the example count. That is our decision rule, unit plus repeat count, with
the cutoff in the same place we calibrated ours to. The DRY sampler in llama.cpp
takes the continuous version, penalising in proportion to the length of the
matched repeated sequence, exponentially in match length. Its fixed-length
ancestor is n-gram blocking, the `no_repeat_ngram_size` family. Grammar-based
structural repetition detection has also been proposed for code specifically.
Suffix automata are used for the same suffix-matching primitive in speculative
decoding, for an unrelated purpose.

**The algorithm is textbook, and should be presented as such.** Longest repeated
substring is a classic string problem, solved with suffix trees or suffix
arrays. Our binary search over lengths with rolling hashes is a standard
equivalent formulation, not a new algorithm. Saying so costs nothing and
prevents a reviewer from reading the pseudocode as a claimed contribution. The
part worth pseudocode is the pipeline around it: digit normalization, the
reduction to a minimal unit, the repeat count, and the mapping back to real
token positions.

**Why LRS earns its place in this paper specifically.** Three reasons, in order
of strength:

1. It is the only structural rule that produces a *position* rather than a
   score, and position is the object the entire evaluation protocol is built
   around. The onset agreement result is the argument: its onset sits within a
   single token of the judge's on median for every build, against 130-280 tokens
   early for the windowed repetition score and for entropy. It is therefore the
   natural independent check on the label.
2. It is the metric prior work uses to *define* the phenomenon, so having it
   makes the corpus comparable to that work even though the population is
   constructed differently.
3. Its median absolute onset error, 29-106 tokens, is one of the two inputs to
   the claim that the reference point is uncertain on the same scale as the
   early-warning window.

**The tolerance limitation now has a named alternative.** The argument for
keeping LRS exact-match rests partly on approximate matching being expensive and
having no natural scale. SpecRA (2025) is a counter-example worth citing rather
than ignoring: it projects the token sequence randomly onto a unit-norm complex
sequence and reads periodicity off the peaks of its FFT autocorrelation, which
costs `O(n log n)` and is explicitly robust to the variations that break exact
matching, with number increments and small spelling changes given as the
examples. That is our own worked example, and it is the same failure mode digit
normalization patches by hand. The honest version of our limitations paragraph
therefore drops the claim that tolerant matching has no tractable formulation,
keeps the two arguments that survive (a tolerance is not a threshold on an
existing score, and a periodicity detector does not return the token position
where the pattern starts, which is the output we need), and names SpecRA as the
alternative for anyone who wants tolerance more than they want an onset. Their
base rate is also a useful comparison point for ours: 813 repetitive samples
found in 1.13 million agent records.

**And the boundary that must be respected.** LRS cannot become the label. The
positioning argument is that an onset defined by a repeated n-gram or by chunk
similarity makes the label and the baseline the same object, which is the design
flaw we are pointing at in the prior work. LRS stays a structural cross-check
and a baseline. The judge stays the arbiter.

---

## 6. Reconciliation checklist

- [ ] Rename the windowed score from TTR to a rep-n-based name, and state the
      relation `r_t = seq-rep-2` over a moving window.
- [ ] Add the offline-label versus online-baseline distinction, in both
      documents, at the point the forward window is defined.
- [ ] Mark every carried-over windowed-repetition number as the offline variant.
- [ ] Do not place labeling-side detection numbers and probe-side budgeted
      numbers in the same table or the same sentence: different populations,
      different threshold rules.
- [ ] State that judge-versus-metric onset disagreement bounds joint error.
- [ ] Add `rep/l` with `l = 128` as a second, literature-standard causal
      baseline.
- [ ] Present the LRS search as a standard formulation of a textbook problem.
- [ ] Cite LoopGuard as concurrent work that labels with the same windowed
      statistic at the same window size, and position our labeling numbers as the
      first validation of a rule that is already deployed on thresholds set by
      inspection.
- [ ] Cite Gopher's repetition filters as the web-scale ancestor of thresholding
      a repetition score, so the score reads as standard practice.
- [ ] Cite vLLM's `repetition_detection` and llama.cpp's DRY sampler where the
      LRS decision rule is introduced.
- [ ] Revise the LRS tolerance argument to name SpecRA and drop the
      no-tractable-formulation claim.
- [ ] Resolve the related-work verification marks with the numbers in Section 7.

---

## 7. Reference list, with what each source actually says

**Metric definitions**

- Li, Galley, Brockett, Gao, Dolan (2016), *A Diversity-Promoting Objective
  Function for Neural Conversation Models*. `distinct-n` = unique n-grams over
  total n-grams.
- Welleck, Kulikov, Roller, Dinan, Cho, Weston (2020), *Neural Text Generation
  with Unlikelihood Training*, ICLR. `seq-rep-n = 1 - |unique n-grams| /
  |n-grams|`, reported mainly at `n = 4`. `rep/l` = fraction of next-token
  (top-1) predictions occurring in the previous `l` tokens, `l = 128`. `wrep/l`
  = the same, excluding repeats that match the ground-truth next token. `uniq` =
  number of distinct next-token predictions over a set.
- Covington, McFall (2010), *Cutting the Gordian Knot: The Moving-Average
  Type-Token Ratio (MATTR)*, Journal of Quantitative Linguistics 17(2). Fixed
  window slid one token at a time, TTR per window, averaged; motivated by
  removing the sample-length confound. Typical windows 50-100.
- Liu et al. (2025), *Code Copycat Conundrum: Demystifying Repetition in
  LLM-based Code Generation*. Uses `rep-n`, `rep-line` (exact repeated lines),
  `sim-line` (Levenshtein similarity above 0.8). Reports `rep-3` of 15-45% for
  models against 3.8% for human code, and that 89.9% of repetitive snippets
  exceeded the token limit. Useful for the code-domain finding and as
  independent support for the token-cap gate.

**Mechanism and mitigation**

- Holtzman, Buys, Du, Forbes, Choi (2020), *The Curious Case of Neural Text
  Degeneration*, ICLR. Figure 4: probability of a repeated phrase rises with
  each repetition, a positive feedback loop, holding for most phrases regardless
  of length or whether the phrase was sampled at random.
- Fu, Lam, So, Shi (2021), *A Theoretical Analysis of the Repetition Problem in
  Text Generation*, AAAI. High-inflow problem; average repetition probability as
  the quantity analysed.
- Xu, Liu, He, Wang, Wu et al. (2022), *Learning to Break the Loop: Analyzing
  and Mitigating Repetitions for Neural Text Generation*. Self-reinforcement
  effect: more repetitions of a sentence in context means higher probability of
  producing it again. Human baseline 0.02% consecutive sentence-level repetition
  in Wikitext-103.
- Xu, Ma, Yu, Cai, et al. (2023), *Look-Back: Improving Text Generation with
  Look-Back Decoding*, EMNLP. KL divergence between the current next-token
  distribution and historical ones, used online to avoid repetition and topic
  drift.
- Li, Lan, Fu, Cai, Liu, Collier, Watanabe, Su (2023), *Repetition In Repetition
  Out: Towards Understanding Neural Text Degeneration from the Data
  Perspective*, NeurIPS. Degeneration correlates with repetition in the training
  data; attention dropout on repetitive training tokens; unifies the earlier
  accounts.
- Dong, Liu, Jiang, Gu, Jin, Li (2025), *Rethinking Repetition Problems of LLMs
  in Code Generation*, ACL, pages 965-985. Structural rather than content repetition; grammar-based
  detection, penalising the tokens that drive it. CodeRepetEval.
- *Repetitions are not all alike: distinct mechanisms sustain repetition in
  language models* (2025), arXiv 2504.01100. Separates in-context-induced from
  natural repetition; natural repetition attends disproportionately to
  low-information tokens and appears to be a fallback when context retrieval
  fails.

**Detecting loops from internal states**

- Duan, Pang, Wei, Duan, Tian, Xu, Deng, Yin, Cheng (2026), *Circular
  Reasoning: Understanding Self-Reinforcing Loops in Large Reasoning Models*,
  arXiv 2601.05693. LoopBench: 700 samples, 7 subtasks, built to induce
  numerical and statement loops. Labels: numerical loop when `k * l > 500`,
  statement loop when `k > 3`, thresholds set empirically. Linear, SVM and MLP
  classifiers on averaged final-layer hidden states reach 0.998 accuracy and
  1.000 AUC for statement loops on the 14B distill. Semantic circularity
  precedes statement repetition. Early prediction with CUSUM: early detection
  rate 0.64-0.76, false positive rate 0.24-0.34, lead time 36.5-51.4 sentences
  or 1305.9-1980.4 tokens; the 7B distill sits at 0.74 EDR with 0.24 FPR.
  Prediction experiments use greedy decoding, balanced test sets of at least 50
  positive and 50 negative cases, and models with fewer than 50 loop cases are
  excluded.
- Xie, Zhang et al. (2025) [first author Wenya Xie], *Word Salad Chopper: Reasoning Models Waste A Ton Of
  Decoding Budget On Useless Repetitions, Self-Knowingly*, EMNLP, arXiv
  2511.00536. Chunks split on blank lines. Label: a chunk is word salad if its
  embedding similarity (all-MiniLM-L6-v2) to any earlier chunk reaches 0.99.
  Single-layer logistic classifier on the final block's hidden state at the
  chunk-final token. Accuracy 89.77-93.52% and AUROC 95.84-98.63 at temperature
  0 across GSM8K, MATH-500, AIME25 and GPQA-Diamond. Their Table 10 tracks one
  example through the loop: classifier score 1.19e-10 at chunk 209, 3.69e-5 at
  chunk 255, 1.000 at chunk 430. That progression is the evidence for reading
  the result as detection strength growing with depth into the loop rather than
  as prediction of onset.
- Yu, Liu, Sun, Shi, Chen (2025), *Breaking the Loop: Detecting and Mitigating
  Denial-of-Service Vulnerabilities in Large Language Models*, arXiv 2503.00416.
  Recurrent generation defined as a subsequence repeated with slight variations
  at a non-overlapping later position, requiring that the generation reach
  `max_new_tokens` (2,000 in their setup). Detector is an MLP over binary
  neuron-activation patterns from MLP sublayers across decoder blocks, using
  sorted maximum token-to-token activation similarities as features: 95.24%
  accuracy, F1 0.87, false positive rate 2.59%, 0.36 ms per call. Inputs that
  trigger loops are found adversarially with an evolutionary search, whose
  fitness function is a self-similarity score computed from output token
  probability distributions. Labels come from the cap plus manual review: they
  inspected the Llama2-7B-chat responses that reached the limit and found all of
  them repetitive, differing mainly in the length of the repeated segment, and
  confirmed this on 800 sampled inputs across eight models.

**Thresholded repetition scores used as labels or filters**

- Xu, Wu, Shi, Cui, Liu, Li, Ma, Liu, Zhu, Xu (2026), *LoopGuard: Breaking
  Self-Reinforcing Attention Loops via Dynamic KV Cache Intervention*, arXiv
  2604.10044. Offline label: `Loop = 1 <=> TTR <= 0.2 and CR <= 0.12 and
  T >= 2480`, where `TTR = |uniq(y_1:T)| / T` and
  `CR = |comp(y_1:T)| / |y_1:T|`, generation cap 2500. Online trigger: sliding
  window W = 256 over the partial output, windowed TTR below 0.2, windowed CR
  below 0.12, and a confidence streak of more than 6 consecutive steps with
  top-1 probability above 0.9, combined by a two-of-three vote, with a 64-token
  warmup and a 32-step cooldown. Reports no detection rate and no lead time.
  Attributes the loop to a subset of attention heads locking onto a narrow
  suffix of the history, stabilised by KV cache reuse. Introduces LoopBench, with
  explicitly loop-inducing conditions.
- Rae et al. (2021), *Scaling Language Models: Methods, Analysis and Insights
  from Training Gopher*, arXiv 2112.11446. MassiveText repetition filters:
  duplicate line fraction 0.30, duplicate line character fraction 0.20,
  duplicate paragraph fraction 0.30, duplicate paragraph character fraction
  0.20, top 2/3/4-gram character fractions 0.20/0.18/0.16, and duplicate n-gram
  character fractions from 0.15 at n = 5 down to 0.10 at n = 10. Documents
  exceeding any threshold are discarded. Inherited by FineWeb, RedPajama-v2 and
  Dolma quality signals, and reimplemented in data-prep-kit's Gopher repetition
  annotator.
- Alibaba PAI, *LLM N-Gram Repetition Filter*. Repetition ratio defined as the
  cumulative frequency of n-grams appearing more than once over the total n-gram
  frequency, computed over a sliding window at configurable n, at character or
  word level; documents outside a user-set range are dropped.
- SpecRA (2025), *Monitor Degenerative Repetition in LLM Agents using Randomized
  FFT*, OpenReview xVO4BqmzVD. Randomly projects the vocabulary onto a unit-norm
  complex sequence and reads periodicity from the peaks of the FFT
  autocorrelation: repetitive text shows strong peaks at the repetition period
  while non-repetitive text behaves like white noise. Explicitly tolerant to
  minor variations such as number increments and small spelling changes, which
  is the case exact matching cannot handle. Taxonomy built from 813 repetitive
  samples found in 1.13 million anonymised agent records. Note: only the
  abstract and public summary were readable, the full paper is behind
  OpenReview's bot check, so the mechanism above should be re-checked before it
  is cited in detail.
- n-gram repetition penalties are also used during RL training on reasoning
  models, applied when a repetition ratio over unique n-grams exceeds a set
  threshold, or as an outright negative reward for repetitive and overlong
  responses. Worth a sentence if we want a training-side citation, but the
  specific implementations need checking before being named.

**Repetition control in deployment**

- vLLM, `repetition_detection` on `SamplingParams` (v0.17.0). Scheduler-level
  stop condition checked after sampling, terminating a request with
  `FINISHED_REPETITION` when a repeated n-gram pattern is found. Parameters:
  `min_pattern_size`, `max_pattern_size` and `min_count`, the required number of
  recurrences, documented with 3 as the example.
- DRY sampling, llama.cpp. Matches the current generation against prior context
  and penalises in proportion to the length of the repeated sequence,
  exponentially in match length, with configurable allowed match length,
  lookback and sequence breakers. Longest-suffix-match repetition detection in
  production.
- n-gram blocking, the `no_repeat_ngram_size` family, as the fixed-length
  predecessor.
- Suffix automata are used for the same longest-suffix-match primitive in
  speculative decoding (SAM-Decoding, arXiv 2411.10666), for an unrelated goal.

**Judge labels**

No prior work was found that labels degeneration with an LLM judge, and none
that asks a judge for a verbatim quote marking the onset. The judge literature
validates verdicts, not positions. The onset-as-quote design and its
server-side re-verification therefore look genuinely new, and the reliability
numbers should be presented as the evidence that a judge-produced *position*
can be made checkable.
