# Baselines in LLM failure monitors and degeneration detection, and what this paper needs

Scope. The paper compares causal linear probes on residual-stream activations against untrained scorers (token cap, LRS, rep/ℓ, windowed seq-rep-2, windowed inverted entropy, and an activation self-similarity scorer that is defined in the appendix but has no column in the baseline table, see the TODO at `paper/main.tex` line 1891). The notes below collect which baselines related work uses, and then map the paper's scorers onto them.

Verification note. Every finding below was read from the primary source in this session unless marked "(not fetched this session)". Some fetches returned only part of a long paper (first 100k characters), so appendix-only baselines may be missing; this is noted where it matters.

## Q1. Is predictive entropy (or windowed entropy) an established degeneration or looping signal? Who uses it?

### Takeaway
Low entropy / high confidence during loops is a well documented correlate of repetition, but no degeneration paper found here uses entropy as a detector or as a compared baseline. Token entropy is, however, a standard baseline in hallucination monitors. So the windowed entropy scorer is best justified by citing the hallucination-monitor baseline (SelfCheckGPT, Obeso et al.) for its form, and Mirostat, Holtzman et al., Xu et al. 2022, Pipis et al. and Duan et al. for why it should track loops.

### Cited Findings
- Mirostat reports that with low top-k / top-p, "perplexity drops significantly with generated text length, which is also correlated with excessive repetitions" (the "boredom trap"), and that "cross-entropy has a near-linear relation with repetition in generated text", nearly independent of the sampling method. Basu, Ramachandran, Keskar, Varshney, arXiv:2007.14966 (ICLR 2021 per my knowledge; the arXiv page lists no venue) — [arXiv](https://arxiv.org/abs/2007.14966)
- Xu et al. 2022 describe the "self-reinforcement effect": "the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence." Learning to Break the Loop, NeurIPS 2022 — [arXiv:2206.02369](https://arxiv.org/abs/2206.02369)
- Holtzman et al. 2020 show that the probability of a repeated phrase increases with each repetition (positive feedback loop). (Figure-level claim from my knowledge of the paper, not fetched this session.) — [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- Pipis et al. 2025 detect loops with an n-gram rule, not entropy. They observe that during a loop the next-token distribution starts out "relatively normal" but "as it repeats the same text, it becomes increasingly confident", and treat this as a catalyst that makes loops hard to escape. No entropy-based detector is evaluated. — [arXiv:2512.12895](https://arxiv.org/html/2512.12895)
- Duan et al. 2026 (Circular Reasoning) report that loop onset brings "a corresponding drop in entropy toward zero" (their Fig. 3), but their detector is a CUSUM change-point test on a linear hidden-state score; entropy is not a compared baseline. They mention high-entropy trigger tokens ("But", "Wait") preceding statement loops. — [arXiv:2601.05693](https://arxiv.org/html/2601.05693)
- Ivgi et al. 2024 frame sequence repetition, degenerate text and hallucination as "fallback behaviors" under uncertainty; within one generation, as uncertainty grows, models shift from hallucination to degenerate text to sequence repetition. NeurIPS 2024 — [arXiv:2407.06071](https://arxiv.org/abs/2407.06071)
- Token-level entropy is a standard monitor baseline in hallucination detection. SelfCheckGPT defines grey-box baselines Avg(−log p), Max(−log p), Avg(H), Max(H) over a sentence (EMNLP 2023) — [ar5iv 2303.08896](https://ar5iv.labs.arxiv.org/html/2303.08896). Obeso et al. 2025 compare their streaming entity-hallucination probes against token-level entropy, token-level perplexity and semantic entropy (linear probe 0.867 and LoRA probe 0.905 AUC vs. 0.71 for the three uncertainty baselines on LongFact, Llama-3.3-70B) — [arXiv:2509.03531](https://arxiv.org/html/2509.03531). Orgad et al. 2025 use logits-mean and logits-min (aggregated token probabilities) and P(True) as baselines — [arXiv:2410.02707](https://arxiv.org/html/2410.02707). Kossen et al. 2024 (semantic entropy probes) compare against naive entropy, length-normalised log-likelihood, P(True), accuracy probes and sampled semantic entropy — [arXiv:2406.15927](https://arxiv.org/html/2406.15927).
- An entropy signal is also used for the opposite purpose. EAT (Entropy After </think>) treats low next-token entropy after an inserted </think> as a sign the model has converged and can exit early. — [arXiv:2509.26522](https://arxiv.org/abs/2509.26522)
- A secondary summary of the looping literature states that loop onset shows a rise in the top-1 logit and entropy collapse (aggregator page, not a primary source) — [emergentmind](https://www.emergentmind.com/topics/looping-in-reasoning-models)

### Inferences
- The paper currently cites `holtzman2020curious` and `pipis2025wait` for windowed entropy. Adding Basu et al. (Mirostat) gives the most direct quantitative link between low cross-entropy and repetition, and Xu et al. 2022 gives the self-reinforcement mechanism. Adding SelfCheckGPT's Avg(H) and Obeso et al.'s token-entropy baseline shows that "mean token entropy over a span" is the standard uncertainty baseline that probe papers compare against. Together these make the scorer a recognised baseline instead of an invented one.
- The windowed mean of H_t over W tokens is SelfCheckGPT's Avg(H) computed over a trailing token window instead of a sentence. The 1/(1+H̄) map is monotone, so it changes nothing for ranking metrics or calibrated thresholds; it is a presentation choice only.
- A cheap sibling worth adding is windowed mean negative log-probability of the sampled token (SelfCheckGPT Avg(−log p), Obeso's "perplexity", Orgad's logits-mean). Reviewers from the hallucination side expect entropy and perplexity to appear together. It costs nothing because logits are already stored.
- Ivgi et al. and the "Wait"/"But" trigger tokens in Duan et al. suggest uncertainty may go up before a loop and down once it starts. This is consistent with the paper's result that windowed entropy is strong at detection (AUC 0.994, Post 0.86 at β=10%) and weak at early warning (Warn 0.083, lead +244). The paper could state this explicitly as the reason inverted entropy is a detection baseline and not an early-warning one.
- EAT shows low entropy also means "converged answer", so a pure low-entropy alarm is expected to fire on healthy, confident text. This supports reporting the false-alarm budget for entropy, which the paper already does.

### Gaps
- I found no paper that uses windowed entropy as a deployed loop detector or as a compared baseline in degeneration detection. The claim "entropy is an established loop detector" cannot be supported; "low entropy is an established loop correlate" can.
- Holtzman's figure claim was not re-read this session.

## Q2. Which repetition statistics are standard and citable as detectors (seq-rep-n, rep/ℓ, distinct-n, LRS, n-gram detectors in practice)?

### Takeaway
seq-rep-n and rep/ℓ (Welleck et al. 2020) and distinct-n (Li et al. 2016) are standard evaluation metrics for degeneration, not online detectors. The detectors used in practice are repeated-n-gram rules (n-gram pattern repeated at least k times), used both as loop labels in recent reasoning-model papers and as an early-stop option in vLLM. LRS has no established use as a detector. Note that Welleck's rep/ℓ is defined on top-1 next-token predictions under ground-truth prefixes, so the paper's version on sampled tokens is an adaptation and should be described as such.

### Cited Findings
- seq-rep-n: 1 minus the number of unique n-grams over the total number of n-grams in a generated continuation, "zero when the continuation has no repeating n-grams, and increases towards 1.0 as the model repeats." — [Welleck et al. 2020, ar5iv 1908.04319](https://ar5iv.labs.arxiv.org/html/1908.04319)
- rep/ℓ: "the fraction of next-token (top-1) predictions that occur in the previous ℓ tokens", averaged over positions, computed with ground-truth prefixes from the validation/test set. wrep/ℓ additionally excludes repeats equal to the ground-truth next token. ℓ=128 is the value used in their Section 4 analysis (next-token predictions occur in the previous 128 words 62% of the time vs. 49% for human text). — [Welleck et al. 2020](https://ar5iv.labs.arxiv.org/html/1908.04319)
- distinct-n (ratio of distinct n-grams to total generated n-grams), Li et al. 2016, NAACL. (Not fetched this session.) — [arXiv:1510.03055](https://arxiv.org/abs/1510.03055)
- Pipis et al. mark a response as looping "if it contains some n-gram appearing at least k times", with n=30 and k=20 for reasoning models and k=10 for instruct models. Looser settings (10-grams, low k) mostly gave false positives on manual check; looping percentages were stable across (n, k). — [arXiv:2512.12895](https://arxiv.org/html/2512.12895)
- Duan et al. define numerical loops by k×l > 500 and statement loops by k > 3 (k repeats of a unit of length l). These are labelling rules, not baselines. — [arXiv:2601.05693](https://arxiv.org/html/2601.05693)
- Word Salad Chopper labels a "\n\n"-delimited chunk as word salad if its MiniLM sentence embedding has cosine ≥ 0.99 with one of the previous 100 chunks, and states that rule-based methods are not applicable because outputs rarely repeat verbatim. — [arXiv:2511.00536](https://arxiv.org/html/2511.00536)
- vLLM `SamplingParams.repetition_detection` takes `RepetitionDetectionParams` ("Parameters for detecting repetitive N-gram patterns in output tokens") with `min_pattern_size`, `max_pattern_size` and `min_count` (must be ≥ 2); if repetition is detected, "generation will be ended early". It is off by default and the full parameter set appears only in recent docs (v0.21+). — [vLLM docs](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/)
- vLLM also has an open feature request for a "ThinkingLoopBreaker" that detects exact periodic repetition inside the reasoning section and forces the end of reasoning. — [vLLM issue #52673](https://github.com/vllm-project/vllm/issues/52673)
- Hugging Face transformers offers `no_repeat_ngram_size` ("all ngrams of that size can only occur once"), applied as a logits processor that bans tokens, not as a stopping criterion. I found no built-in repetition-based stopping criterion. — [transformers docs](https://huggingface.co/docs/transformers/v5.13.0/main_classes/text_generation)
- DRY (Don't Repeat Yourself) is a sampling-time penalty on tokens that would extend a suffix already seen earlier in the context, with a penalty that grows with match length; it is implemented in llama.cpp, text-generation-webui and ExLlamaV2. Its paper compares against repetition/presence/frequency penalties, no-repeat n-gram blocking, contrastive decoding and a placebo. arXiv:2608.22761 (Aug 2026; read via alphaXiv) — [alphaXiv](https://www.alphaxiv.org/abs/2608.22761.md)
- Pangu Embedded describes repetition flagging by Jaccard similarity over recent n-grams in a local window, checked periodically. (Search snippet only.) — [arXiv:2505.22375](https://arxiv.org/pdf/2505.22375)
- vLLM's `repetition_penalty`, `presence_penalty` and `frequency_penalty` penalise tokens already seen; they are mitigations, not detectors. — [vLLM docs](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/)

### Inferences
- The paper's windowed seq-rep-2 is Welleck's seq-rep-2 restricted to a trailing window. This is a clean, citable baseline and should stay.
- The paper's rep/ℓ indicator (current token appeared in the previous ℓ=128 tokens) is Welleck's rep/ℓ with the model's top-1 prediction under gold context replaced by the sampled token in the model's own generation. The text should say "adapted from rep/ℓ" and give the reason (no gold continuation exists at deployment). ℓ=128 matches the value in Welleck's Section 4.
- seq-rep-2 and rep/ℓ are both lexical repetition rates over 128 tokens; reviewers may see one of them as redundant. Keeping both is fine if the text notes that rep/ℓ is a unigram and seq-rep-2 a bigram statistic.
- LRS is the weakest link. It is non-causal (needs the full response), has no detector precedent, and fails every budget in Table tab:baselines-vs-probes (†). The natural, citable replacement is a causal repeated-n-gram detector of the kind shipped in vLLM and used as the label rule in Pipis et al.: alarm at the first position where some n-gram of length in [n_min, n_max] has occurred at least k times in the prefix. Its score can be the maximum count of any n-gram in the trailing window. This is what a practitioner would actually turn on, so it is the baseline a reviewer from the systems side would ask for. Manber and Myers (suffix arrays) can remain as the algorithmic citation if LRS is kept as a non-causal oracle, but it should then be labelled as an oracle, not a monitor.
- None of the degeneration papers (Duan, Word Salad Chopper, Yu et al.) compare against any other loop detector, so there is no established baseline set to match in this subfield. The defensible position is to tie each lexical baseline to (a) the standard metric it comes from (Welleck) and (b) the detector used in practice (vLLM / Pipis rule).

### Gaps
- vLLM docs do not state the finish reason returned when repetition detection fires; I did not read the source.
- I did not confirm whether llama.cpp or SGLang have a detection (stopping) mode separate from DRY's penalty.

## Q3. Do probe papers compare to text classifiers fine-tuned on partial outputs, or to an LLM judge reading the prefix? Would a reviewer expect a text-only classifier on the prefix?

### Takeaway
Yes. Almost every activation-monitor paper in the safety and hallucination literature includes a black-box text baseline: a prompted LLM, a fine-tuned text classifier, or the model's own self-evaluation. Streaming harmful-content work evaluates moderators on cumulative prefixes and trains text classifiers on partial outputs. A reviewer is likely to ask whether activations carry information about upcoming degeneration that the text prefix does not, and the current baselines cannot answer this because they are all untrained.

### Cited Findings
- McKenzie et al. 2025 compare attention probes against prompted LLM classifiers (Llama-3.2-1B, Llama-3.1-8B, Llama-3.3-70B, Gemma-3-1B/12B/27B; best of four prompts on dev) and fully fine-tuned LLM classifiers (unembedding removed). Probes are "comparable to that of prompted or finetuned medium-sized LLM monitors"; at 1% FPR the best probe detects 43% of high-stakes cases vs. 55% for the best LLM baseline; a probe + LLM cascade beats both. — [arXiv:2506.10805](https://arxiv.org/html/2506.10805)
- Goldowsky-Dill et al. 2025 compare deception probes against a black-box GPT-4o reading the complete transcript (7-point scale), and calibrate the probe's 1% FPR threshold on Alpaca control responses. — [arXiv:2502.03407](https://arxiv.org/html/2502.03407)
- Obeso et al. 2025 include a black-box self-evaluation baseline (the model is prompted to judge whether a sentence of its own output contains a hallucination). — [arXiv:2509.03531](https://arxiv.org/html/2509.03531)
- Azaria and Mitchell 2023 (SAPLMA) compare their hidden-state classifier against few-shot prompting and a BERT classifier trained on the statement text with the same architecture. — [ar5iv 2304.13734](https://ar5iv.labs.arxiv.org/html/2304.13734)
- Li et al. 2025 (FineHarm / streaming content monitor) compare against full-detection moderators (HateBERT, ToxDectRoBERTa, LlamaGuard-3-8B, Perspective, OpenAI Moderation) and fine-tuned ModernBERT and Qwen2.5 (0.5B–7B) in full and partial variants. Partial baselines run full detection on each cumulative prefix; token predictions are aggregated with a Delay-k rule (stop after k flagged tokens, k tuned on validation F1). Their monitor reads about 18% of tokens on average. — [arXiv:2506.09996](https://arxiv.org/html/2506.09996)
- Qwen3Guard-Stream adds a token-level classification head for moderating output during incremental generation (0.6B/4B/8B). — [arXiv:2510.14276](https://arxiv.org/pdf/2510.14276v1); [HF model card](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B)
- Constitutional Classifiers++ trains linear probe classifiers and ensembles them with external classifiers in a cascade to cut cost (arXiv:2601.04603; author attribution to Cunningham et al. comes from a secondary page and was not confirmed). — [arXiv abs](https://arxiv.org/abs/2601.04603)
- Word Salad Chopper does not compare its final-layer linear classifier with any other detector; its baselines are decoding penalties (repeat/presence/frequency) and brute-force truncation. — [arXiv:2511.00536](https://arxiv.org/html/2511.00536)
- Duan et al. compare only internal ablations (no persistence p=1; linear vs. SVM vs. MLP for separability). — [arXiv:2601.05693](https://arxiv.org/html/2601.05693)

### Inferences
- The two closest degeneration papers set a low bar (no external detector), but the paper presents itself as a monitor paper and will be read by people who know McKenzie et al. and Goldowsky-Dill et al. Those readers expect a black-box text baseline.
- The cheapest defensible text baseline is a small fine-tuned text classifier (e.g. ModernBERT or a ~0.5B LM with a classification head) that reads the trailing window of generated tokens (e.g. last 512 tokens) and is trained with the same onset-shifted labels, splits and step budget as the probe. Evaluated with the same calibrated alarm rule, it answers directly whether the probe's early-warning advantage comes from the activations or from surface text. FineHarm's partial-output setup is the citation for this design.
- An LLM judge on prefixes is the natural second option because the paper already uses an LLM judge to label onsets. Asking it, on a prefix ending at t, "is this response degenerating or about to?" gives a black-box monitor with no training. It is expensive per token, so it could be evaluated at a stride (e.g. every 64 tokens) on the test split only. This is optional for a workshop paper.
- If neither is feasible, the limitations section should state that the comparison is only against untrained scorers, so the result shows that the probe beats lexical and confidence heuristics, not that activations beat text.

### Gaps
- McKenzie et al. Appendix B.8 ("Chain-of-Thought Baselines") was truncated in the fetch.
- I found no degeneration-specific paper that trains a text classifier on prefixes.

## Q4. Is a final-layer probe or a logit-lens readout a standard baseline against middle-layer probes?

### Takeaway
Layer sweeps that include the final layer are standard, and several papers report middle layers as best. A logit-lens or tuned-lens readout is not a standard baseline in any monitor paper found here. For this paper the per-layer sweep already covers the final-layer probe, and the final-layer logit lens is exactly the model's output distribution, which the windowed-entropy baseline already uses.

### Cited Findings
- SAPLMA probes the last, 28th, 24th, 20th and 16th layers of 32-layer models; the 20th (OPT-6.7b) and middle 16th (Llama2-7b) work best on the true-false set, the 28th on self-generated sentences. — [ar5iv 2304.13734](https://ar5iv.labs.arxiv.org/html/2304.13734)
- Semantic entropy probes are trained at all layers; AUROC increases toward later layers for short-form and peaks at intermediate layers for long-form generation. — [arXiv:2406.15927](https://arxiv.org/html/2406.15927)
- Orgad et al. choose the probing layer per token position by validation performance. — [arXiv:2410.02707](https://arxiv.org/html/2410.02707)
- Goldowsky-Dill et al. sweep layers and use layer 22 of Llama-3.3-70B, chosen on the Roleplaying dataset. — [arXiv:2502.03407](https://arxiv.org/html/2502.03407)
- Obeso et al. attach the probe at layer ⌊0.95 × L⌋. — [arXiv:2509.03531](https://arxiv.org/html/2509.03531)
- Word Salad Chopper uses the output of the final transformer block at "\n\n" tokens, without a stated reason beyond the delimiter choice. — [arXiv:2511.00536](https://arxiv.org/html/2511.00536)
- Constitutional Classifiers++ ablations find performance degrades as fewer layers are used for probing (search snippet). — [arXiv:2601.04603](https://arxiv.org/pdf/2601.04603)
- The tuned lens trains an affine map per layer to decode hidden states into vocabulary distributions and is reported as more reliable than the logit lens; the authors use latent prediction trajectories to detect malicious inputs. — [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)

### Inferences
- Reporting the final-layer probe from the existing layer sweep, next to layer 15, is enough to answer "why not the last layer". Word Salad Chopper probes the final block, so this comparison also positions the paper against its closest prior detector.
- A logit-lens baseline would add little. At the last layer it equals the model's output distribution, whose entropy is already the windowed-entropy baseline. At middle layers, logit-lens entropy is an untrained intermediate-layer signal and could be a one-line extension, but no reviewer is likely to require it.

### Gaps
- I did not find any monitor paper that uses logit lens or tuned lens readouts as a compared baseline.

## Q5. What baselines do the closest works use (Xie et al. 2025; Duan et al. 2026; McKenzie et al. 2025; Obeso et al. 2025; Goldowsky-Dill et al. 2025), and which probe-side baselines are standard?

### Takeaway
The degeneration papers (Xie, Duan, Yu) use no external detector baselines. The monitor papers (McKenzie, Obeso, Goldowsky-Dill) use black-box LLM or text baselines and, for hallucination, token entropy, perplexity and semantic entropy. On the probe side, logistic regression is the default; difference-of-means, SAE-feature probes and aggregation variants appear as ablations, and random-label control tasks are a classic probing check that the deception paper does not use.

### Cited Findings
- Xie et al. 2025 (Word Salad Chopper, EMNLP 2025): final-layer linear classifier at "\n\n" tokens; labels from MiniLM chunk similarity; baselines are decoding penalties and brute-force truncation; metrics are classifier accuracy/AUROC, task accuracy, output length, latency. — [arXiv:2511.00536](https://arxiv.org/html/2511.00536)
- Duan et al. 2026: CUSUM on a linear hidden-state score; compared only with its own p=1 ablation ("marginally higher recall" but "prohibitive False Positive Rates"); metrics include early detection rate, FPR, average sentence and token earliness. — [arXiv:2601.05693](https://arxiv.org/html/2601.05693)
- Yu et al. 2025: MLP classifier on MLP-neuron activation similarity patterns; states "no existing baselines address recurrent generation in LLMs"; only baseline is random sampling for their input generator. — [arXiv:2503.00416](https://arxiv.org/html/2503.00416)
- McKenzie et al. 2025: prompted and fine-tuned LLM classifiers (1B–70B); probe aggregations mean, max, last token, max of rolling means, softmax, attention. — [arXiv:2506.10805](https://arxiv.org/html/2506.10805)
- Obeso et al. 2025: token entropy, token perplexity, semantic entropy, black-box self-evaluation; linear vs. LoRA probes. — [arXiv:2509.03531](https://arxiv.org/html/2509.03531)
- Goldowsky-Dill et al. 2025: GPT-4o black-box, SAE-activation probe, follow-up-question probe, descriptive-question probe; fitting methods difference of means, LDA and LAT vs. logistic regression (LR best at separating deceptive from control responses); layer, regularisation and aggregation (mean, max, last token) sweeps; Alpaca control for 1% FPR; no random-direction control. — [arXiv:2502.03407](https://arxiv.org/html/2502.03407)
- Mass-mean (difference-of-means) probing: Marks and Tegmark report it performs about as well as logistic regression, generalises well out of distribution and is more causally implicated in model outputs. — [arXiv:2310.06824](https://arxiv.org/pdf/2310.06824)
- Control tasks: Hewitt and Liang assign random outputs to word types and define selectivity as task accuracy minus control accuracy; linear probes are more selective than MLPs. — [arXiv:1909.03368](https://arxiv.org/pdf/1909.03368)
- SAE probes: Kantamneni et al. (ICML 2025) find SAE-feature probes occasionally win on single datasets but do not consistently beat logistic regression and other baselines. — [PMLR v267](https://proceedings.mlr.press/v267/kantamneni25a.html); [arXiv:2502.16681](https://arxiv.org/abs/2502.16681)
- Orgad et al. compare probes at different token positions (last generated token, token before it, end of question, exact answer token). — [arXiv:2410.02707](https://arxiv.org/html/2410.02707)

### Inferences
- A difference-of-means probe at the same layer is a one-line, training-free ablation that reviewers from the probing side know. It would show whether the gain needs a trained direction. Optional.
- A label-shuffled control (e.g. onsets permuted across responses, or a probe trained on labels from a random onset) is the probing-standard sanity check for the early-warning claim specifically, since it tests whether pre-onset signal is real or a positional artefact. The logbook mentions a positional control already exists; if so it should be framed with the Hewitt and Liang citation.
- SAE probes and attention-head probes are not needed. SAE probes do not reliably beat LR (Kantamneni), and McKenzie's attention aggregation targets response-level classification, not per-token alarms.

### Gaps
- Goldowsky-Dill and Obeso appendices beyond the first 100k characters were not read, so smaller ablations may be missing from this list.

## Q6. Assessment for this paper: which baselines are standard, which of the authors' map to established measures, which are unjustified, and a minimal defensible set

### Takeaway
Three of the paper's scorers map onto established measures (windowed seq-rep-2, rep/ℓ, windowed entropy) and only need better citations and wording. LRS is not justified as a monitor and should be replaced by a causal repeated-n-gram detector matching vLLM practice. The token cap is a reference point, not a baseline. The main missing piece is a learned text-only classifier on the prefix.

### Cited Findings
- Mapping of the paper's scorers (definitions in `paper/main.tex` lines 916–1066):
  - Windowed seq-rep-2 = Welleck seq-rep-n with n=2 over a trailing 128-token window — [Welleck et al. 2020](https://ar5iv.labs.arxiv.org/html/1908.04319)
  - rep/ℓ (ℓ=128, smoothed over 128) = Welleck rep/ℓ adapted from top-1 predictions under gold prefixes to sampled tokens — [Welleck et al. 2020](https://ar5iv.labs.arxiv.org/html/1908.04319)
  - Windowed entropy = SelfCheckGPT Avg(H) over a trailing token window; same family as Obeso et al.'s token-entropy baseline; repetition link from Mirostat and Xu et al. — [SelfCheckGPT](https://ar5iv.labs.arxiv.org/html/2303.08896); [Obeso et al.](https://arxiv.org/html/2509.03531); [Mirostat](https://arxiv.org/abs/2007.14966); [Xu et al. 2022](https://arxiv.org/abs/2206.02369)
  - LRS: no detector precedent; nearest practical analogue is vLLM's n-gram repetition detection and the Pipis n-gram rule — [vLLM docs](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/); [Pipis et al.](https://arxiv.org/html/2512.12895)
  - Token cap: corresponds to stopping at max_tokens (Word Salad Chopper's brute-force truncation is the closest compared baseline) — [arXiv:2511.00536](https://arxiv.org/html/2511.00536)
  - Activation self-similarity: related to Yu et al.'s activation-similarity features (they train an MLP on them; the paper uses similarity directly) — [arXiv:2503.00416](https://arxiv.org/html/2503.00416)

### Inferences
Recommended minimal set, with one-line definition, citation, and need:

| Baseline | Definition | Citation | Needed? |
|---|---|---|---|
| Never alarms / token cap | Reference scorers: no alarm, or alarm on the last token of a capped response | inference-engine max_tokens; Word Salad Chopper truncation | Keep as reference rows. They show AUC saturation, which is the paper's main point. Present them as reference points, not competitors. |
| Windowed seq-rep-n | 1 − unique n-grams / n-grams in the trailing W tokens | Welleck et al. 2020 | Keep. Standard degeneration metric, cheap, causal. |
| rep/ℓ (adapted) | Fraction of the last W sampled tokens that appeared in their preceding ℓ tokens | Welleck et al. 2020 (rep/ℓ), adapted to sampled tokens | Keep, but say "adapted" and why. Could be merged with seq-rep-n if space is tight. |
| Causal repeated-n-gram detector | Alarm when some n-gram (n in [n_min, n_max]) has occurred ≥ k times in the prefix; score = max n-gram count in window | vLLM RepetitionDetectionParams; Pipis et al. 2025 rule; Duan et al. labelling rule | Add, replacing LRS. This is what practitioners deploy, and the reviewer will ask why the probe beats it. |
| Windowed entropy | Mean next-token entropy of the unfiltered distribution over the trailing W tokens, inverted | SelfCheckGPT Avg(H); Obeso et al.; Mirostat; Xu et al. 2022 | Keep. It is the strongest baseline and must stay. Add the hallucination-baseline and Mirostat citations. |
| Windowed mean −log p (optional) | Mean negative log-probability of sampled tokens over the trailing window | SelfCheckGPT Avg(−log p); Obeso "perplexity"; Orgad logits-mean | Optional but nearly free; expected next to entropy by hallucination readers. |
| Learned text-only classifier on prefix | Small encoder or LM fine-tuned on the trailing window of generated text with the same labels and alarm rule | FineHarm partial-output baselines; McKenzie fine-tuned classifiers; SAPLMA BERT baseline | The main missing baseline. Needed to support any claim that activations give earlier warning than text. If omitted, state it as a limitation. |
| LLM judge on prefix (optional) | The labelling judge prompted on prefixes at a stride | McKenzie prompted classifiers; Goldowsky-Dill GPT-4o baseline | Optional for a workshop paper; costly. |
| Final-layer probe | Same probe at the last layer | SAPLMA, SEP layer analyses; Word Salad Chopper uses final block | Already in the layer sweep; just report it next to layer 15. |
| Difference-of-means probe (optional) | w = mean(h | pos) − mean(h | neg), no training | Marks and Tegmark 2023; Goldowsky-Dill App. D.1 | Optional ablation. |
| Label-shuffled / positional control | Probe trained on onsets permuted across responses | Hewitt and Liang 2019 control tasks | Recommended if the early-warning claim is central; cite Hewitt and Liang for any existing positional control. |
| Activation self-similarity | Max cosine to earlier activations in window, lag ≥ λ | Yu et al. 2025 (related) | Either add its column (TODO in Table) or remove the definition. Not required by reviewers. |

Not needed: semantic entropy and P(True) (they need sampled continuations or prompting and target factual correctness, not loops), SAE-feature probes, attention-head probes, logit/tuned lens readouts.

### Gaps
- Whether a text-only classifier closes the early-warning gap is an empirical question this research cannot answer; it decides how strong the paper's claim can be.
- I did not find a published degeneration-detection benchmark with an agreed baseline set, so "standard" here means standard in adjacent monitor literature.
