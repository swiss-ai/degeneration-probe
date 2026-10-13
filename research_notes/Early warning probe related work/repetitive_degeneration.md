# Repetitive degeneration in LLM generation: characterisation, mechanisms, detection and mitigation

Verification note. Every arXiv id below was checked against the arXiv API on 2026-10-12 (title, first author, first-submission date). Venues come from the arXiv comment field, the venue page, or ACL Anthology. Items marked [UNVERIFIED] were only seen in a secondary source (mostly the survey arXiv 2603.16068) and their ids were not confirmed. Items marked [abstract only] were checked only at abstract level.

Entry format used throughout: **Paper** (id, venue) | signal | online? | granularity | metrics | models.

## Q1. Which papers detect or predict degeneration from hidden states, and do any predict the onset before repetition appears in the text?

### Takeaway
At least six works read internal states to detect looping, and three of them claim a signal before textual repetition starts (Duan et al. 2026 with last-layer sentence states plus CUSUM, RADAR 2026 with attention statistics, and a weak single-model probe of budget non-convergence by Oladri et al. 2026). None of them evaluates at token level against an annotated onset with a lead-time-versus-false-alarm protocol on non-reasoning instruct models such as Llama-3.1-8B or Mistral-7B, and none uses an LLM judge to locate the onset.

### Cited Findings

Internal-state detectors and predictors (closest prior work)

- **Word Salad Chopper, WSC** (Xie, Zhong, Le, Xu, Xie, Liu; arXiv 2511.00536, EMNLP 2025 main, pp. 33588-33598) | signal: output of the final transformer block at the "\n\n" token that ends each reasoning chunk | online, chops after two consecutive flagged chunks | chunk level | classifier accuracy and AUROC (AUROC 95.8-98.6 on DeepSeek-R1-Distill-Qwen-7B), plus task accuracy and length savings | DeepSeek-R1-Distill-Qwen-1.5B/7B, DeepSeek-R1-Distill-Llama-8B, Qwen3-8B thinking mode (accuracy about 78-88%) — [arXiv HTML](https://arxiv.org/html/2511.00536v1); [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1705/)
- **Circular Reasoning / LoopBench / CUSUM** (Duan, Pang, Wei, et al.; arXiv 2601.05693, Jan 2026, preprint) | signal: linear, SVM and MLP classifiers on hidden states separating loop from normal states (accuracy about 0.99 on DS-Qwen-14B and Qwen3-8B); the early-warning part uses the mean of each sentence's last-layer hidden states, a linear score x_i = w^T h_i + b, and a CUSUM statistic with a persistence rule | online, sentence level | Early Detection Rate (EDR, about 0.64-0.76), FPR (about 0.24-0.34 with persistence, 0.44-0.62 without), Average Sentence/Token Earliness; reports an average lead of over 40 sentences (about 1500 tokens) before textual repetition | many reasoning, base and instruct models, including Llama-3.1-8B-Instruct and Qwen2.5-Instruct for loop-rate measurement; mechanism analysis only on DS-Qwen-14B and Qwen3-8B; greedy decoding — [arXiv HTML](https://arxiv.org/html/2601.05693v1); [arXiv](https://arxiv.org/abs/2601.05693)
- **RecurrentDetector** in "Breaking the Loop" (Yu, Liu, Sun, Shi, Chen; arXiv 2503.00416, preprint, cs.CR) | signal: binarised MLP neuron activation states of all decoder layers; for each token the maximum fraction of identical binary entries with any other token, sorted and unsorted vectors fed to a 3-layer MLP (d = 120) | online, starts at 400 generated tokens and updates each token | response level (is this generation recurrent) | accuracy 95.24%, F1 0.87, FPR 2.59%, recall 82.88%, 0.36 ms per call | Llama3-8B, Llama2-7B/13B, Gemma2-2B, Vicuna-v1.5-7B/13B. A sample counts as recurrent only if it hits max_new_tokens (2000), the same "runs to the cap" label as our setting, but loops are triggered by adversarial prompts found with an evolutionary search (RecurrentGenerator) — [arXiv HTML](https://arxiv.org/html/2503.00416v1)
- **RADAR** "When Reasoning Goes Astray: Attention Dynamics of Uncontrolled Reasoning" (Zhang, Wang, Ren, Gao, et al.; arXiv 2609.38817, Sep 2026, preprint) | signal: attention statistics averaged over all layers (Prompt Attention Share slope, Cross-context Token Similarity) plus log progress, fed to a 4-class linear softmax (direct, effective reflection, excessive reflection, persistent looping) | online at geometrically spaced checkpoints (2^n tokens) | prefix level | micro-F1 89.8%; reports the token position at which harmful-state confidence crosses 50/70/90%, and says the 70% threshold is crossed before loop onset for naturally induced attacks (for example, Recur attack: onset at 994 tokens, 50% at 75, 70% at 267) | DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-14B, QwQ-32B, Qwen-3.6-27B, GLM-4.7-Flash; loops are mostly attack-induced; looping label = hits cap and relative n-gram repetition score >= 2 — [arXiv HTML](https://arxiv.org/html/2609.38817v1)
- **Token Budget Saturation and Mechanistic Early Detection of Reasoning Non-Convergence** (Oladri, Jawahar, Mohamed; arXiv 2607.21433, Jul 2026, preprint) | signal: logistic-regression probe on residual hidden states at fixed token positions 50-300; best layer 20 of 28 at token 150 | prefix level, one prediction per position | AUC 0.608 +/- 0.080 (5-fold CV), sweep-level permutation p = 0.063; beats a behavioural baseline of entropy and bigram/trigram repetition rates by +0.035 AUC on average | DeepSeek-R1-Distill-Qwen-7B only, greedy, 200 AIME problems, 10k-token ceiling. Target is "never emits </think> within budget", which the authors describe as loops but do not measure directly — [arXiv HTML](https://arxiv.org/html/2607.21433v1); related GitHub repo [oladri-renuka/early_detection](https://github.com/oladri-renuka/early_detection)
- **LoopGuard** (Xu, Wu, Shi, et al.; arXiv 2604.10044, Apr 2026, preprint) [abstract only] | signal: attention heads that lock onto a narrow suffix of the history, read through the KV cache | "detects loop onset online", then prunes repetitive tail spans of the KV cache | reports loop incidence reduced by over 90 percentage points on its own LoopBench | models not named in the abstract — [arXiv](https://arxiv.org/abs/2604.10044)
- **Qwen-Scope** (Deng, Wang, Wang, Wan, et al.; arXiv 2605.11887, May 2026, tech report) | SAE features on Qwen3/Qwen3.5 residual streams whose activation jumps and stays high during endless repetition and stays near zero in normal responses; suppressing them lowers repetition, amplifying them induces it; the signal is used in SFT/RL objectives, not as an early-warning detector — [arXiv](https://arxiv.org/abs/2605.11887); [Alibaba Cloud blog](https://alibabacloud.com/blog/qwen-scope-decoding-intelligence-unleashing-potential_603083)
- **Reasoning Models Know When They're Right** (Zhang, Chen, Pan, et al.; arXiv 2504.05419) [abstract only] | probes on hidden states predict correctness of intermediate answers and are used for early exit from reasoning; this is the "probe-based early-halt" method WSC cites (as Zhang et al. 2025a). Target is correctness, not looping — [arXiv](https://arxiv.org/abs/2504.05419); citation context in [WSC](https://arxiv.org/html/2511.00536v1)

Mechanistic work that locates repetition in internal states but does not build a predictor

- **Repetition Neurons** (Hiraoka, Inui; arXiv 2410.13497, NAACL 2025 short) | neurons found by the difference of FFN activations in the 30 tokens after versus before the onset; onset is detected automatically (a 10-token sequence recurring three times at equal intervals within 100 tokens) | most neurons in the final layer, a second cluster in middle layers ("detectors" vs "replicators") | deactivation after onset cuts repetitive samples by up to about 25-35% | Gemma-2B, Pythia-2.8B, LLaMA-3.2-3B, LLM-jp-3-1.8B. No detection or prediction use — [arXiv HTML](https://arxiv.org/html/2410.13497v2); [ACL Anthology](https://aclanthology.org/2025.naacl-short.41/)
- **Repeat Curse / "Duplicatus Charm"** (Yao, Yang, Xu, et al.; arXiv 2504.14218, Findings of ACL 2025) | finds layers most involved in repetition via logit analysis, then SAE "repetition features"; deactivating them reduces repetition — [arXiv](https://arxiv.org/abs/2504.14218v3); [papers.cool ACL listing](https://papers.cool/venue/2025.findings-acl.406@ACL)
- **Scaling and evaluating sparse autoencoders** (Gao et al., OpenAI; arXiv 2406.04093) reports a dense, recurring latent whose top activations are highly repetitive sequences — [arXiv](https://arxiv.org/pdf/2406.04093)

### Inferences
- The project's description of Xie et al. 2025 and Duan et al. 2026 as probes "that read the final layer" is correct for WSC and for Duan's early-warning score. Duan's main classifier table does not state its layer in the visible text.
- The closest prior claims of "warning before textual repetition" are Duan (sentence level, CUSUM, about 1500 tokens lead, FPR 0.24-0.34) and RADAR (attention features, checkpoint level, attack-induced loops). Neither uses a residual-stream probe across layers, neither reports lead time against an annotated onset at token resolution, and both focus on reasoning models.
- RecurrentDetector shares the "loop that runs to max_new_tokens" label, uses all-layer activations, and runs online, so it should be cited as an internal-state detector. Its loops are adversarially triggered, its detection starts after 400 tokens, and its features measure activation self-similarity, so it detects an ongoing loop and is not an onset predictor.
- Oladri et al. 2026 is the nearest match to "probe residual activations early to predict budget exhaustion", and their weak AUC (about 0.61 at token 150, one model) is a useful contrast point.
- A middle-layer probe would be consistent with Hiraoka and Inui's finding of middle-layer "detector" neurons, while final-layer readers (WSC, Duan) may mostly see the "replicator" stage. This is a hypothesis, not a published result.

### Gaps
- LoopGuard's exact onset signal, models and lead time were not checked beyond the abstract.
- RecurrentDetector's binarisation threshold and training hyperparameters are not reported in the paper.
- Duan's appendix (C.2 onward), including how "Repetition Onset" windows are chosen and the max generation length, was not read in full.
- No paper was found that probes Llama-3.1-8B-Instruct, Mistral-7B-Instruct or Apertus-8B residual streams specifically for loop onset.

## Q2. What standard repetition metrics exist, and where are they defined?

### Takeaway
seq-rep-n, rep/l, wrep/l and uniq come from Welleck et al. 2020; rep-w and rep-r are usually attributed to Fu et al. 2021 together with Welleck; Repetition % and the use of Self-BLEU for degeneration come from Holtzman et al. 2020; Self-BLEU itself is from Texygen (Zhu et al. 2018) and distinct-n from Li et al. 2016. I found no standard, citable definition of a "longest repeated substring" metric.

### Cited Findings
- **seq-rep-n** = 1 - |unique n-grams| / |n-grams| in a continuation, averaged over continuations; **uniq-seq** = number of distinct tokens across continuations; **rep/l** = fraction of argmax next-token predictions that already occur in the previous l tokens; **wrep/l** = same but excluding cases where the repeated prediction equals the ground-truth token; **uniq** = number of distinct argmax predictions. Setup: Wikitext-103, 16-layer transformer, prefix 50, continuation 100, greedy/beam/top-k/nucleus — Welleck, Kulikov, Roller, Dinan, Cho, Weston, "Neural Text Generation with Unlikelihood Training", ICLR 2020, [arXiv 1908.04319](https://ar5iv.labs.arxiv.org/html/1908.04319)
- **Repetition (%)** = share of generations in which "a phrase (minimum length 2) ... repeats at least three times at the end of the generation", measured in the first 200 tokens; the same paper reports **Self-BLEU4**, **Zipf coefficient**, **HUSE** and perplexity, with GPT-2 Large and 200-token generations. It also states "The probability of a repeated phrase increases with each repetition, creating a positive feedback loop" (Figure 4) — Holtzman, Buys, Du, Forbes, Choi, "The Curious Case of Neural Text Degeneration", ICLR 2020, [arXiv 1904.09751](https://ar5iv.labs.arxiv.org/html/1904.09751)
- **Self-BLEU** originates in Texygen — Zhu et al. 2018, [arXiv 1802.01886](https://arxiv.org/abs/1802.01886), cited for Self-BLEU by [Holtzman et al.](https://ar5iv.labs.arxiv.org/html/1904.09751)
- **distinct-n** originates in Li, Galley, Brockett, Gao, Dolan, "A Diversity-Promoting Objective Function for Neural Conversation Models", NAACL 2016 — [arXiv 1510.03055](https://arxiv.org/abs/1510.03055) (definition, distinct n-grams divided by total generated n-grams/tokens, given from knowledge of the paper; not re-read)
- **rep-n** (1 - unique n-grams / (L - n + 1)), **rep-w** (word-level repetition within a prefix window of w = 16) and **rep-r** (portion of repetitive snippets measured by sentence length); Li et al. attribute rep-w and rep-r to Welleck et al. 2020 and Fu et al. 2021 and rep-n to Fu, Welleck, Su and Xu 2022, Su and Collier 2023 — Li, Lan, Fu, et al., "Repetition In Repetition Out", NeurIPS 2023, [arXiv 2310.10226](https://arxiv.org/html/2310.10226v1)
- **Average Repetition Probability (ARP)**, a theoretical quantity with upper bounds derived from a Markov generation model; also the "high inflow" explanation — Fu, Lam, So, Shi, "A Theoretical Analysis of the Repetition Problem in Text Generation", AAAI 2021, [arXiv 2012.14660](https://arxiv.org/pdf/2012.14660); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17520)
- **Repetition-4** (duplicate 4-grams) and **Repetition-Sen** (1 - unique sentences / total sentences), plus consecutive-repetition ratios at word, phrase and sentence level; self-reinforcement measures **TP** (average token probability of the n-th repeat), **IP** (fraction of tokens whose probability increased over the first occurrence), **WR** (winner rate) — Xu, Liu, Yan, Cai, Li, Li, "Learning to Break the Loop" (DITTO), NeurIPS 2022, [arXiv 2206.02369](https://ar5iv.labs.arxiv.org/html/2206.02369)
- **MAUVE** is the quality metric commonly paired with repetition metrics (used by DITTO) — Pillutla et al., NeurIPS 2021, [arXiv 2102.01454](https://arxiv.org/abs/2102.01454); usage in [DITTO](https://ar5iv.labs.arxiv.org/html/2206.02369)
- Loop-specific metrics in reasoning-model work: Numerical Loop Rate and Statement Loop Rate (numerical loop if k*l > 500 tokens; statement loop if a sentence unit recurs more than 3 times) and EDR/FPR/earliness in tokens and sentences — [Duan et al. 2026](https://arxiv.org/html/2601.05693v1); chunk-level "word salad" via MiniLM embedding cosine >= 0.99 against the previous 100 chunks — [WSC](https://arxiv.org/html/2511.00536v1); relative n-gram repetition score rho >= 2 (n = 2-4) plus hitting the cap — [RADAR](https://arxiv.org/html/2609.38817v1)
- Code-specific metrics rep-3, rep-line and sim-line (Liu et al., "Code Copycat Conundrum", 19 code LLMs, 20-pattern taxonomy) — [arXiv 2504.12608](https://arxiv.org/abs/2504.12608)
- RAP, "a metric for balancing repetition and performance" for tuning repetition penalties (Huang et al. 2025a) [UNVERIFIED] — cited in [survey 2603.16068](https://arxiv.org/html/2603.16068v3)

### Inferences
- Holtzman's Repetition % ("ends in a phrase repeated at least three times") is the closest classic metric to the project's response-level label (a loop that runs to the cap). Welleck's rep/l is the classic token-level metric, but it measures next-token copying, not loop membership.
- None of the classic metrics encodes an onset position or lead time. Onset/earliness metrics only appear in 2024-2026 work (Hiraoka and Inui's automatic onset, Duan's ASE/ATE, RADAR's confidence-crossing positions).

### Gaps
- I did not locate the original sentence-level definition of rep-r in Fu et al. 2021; Li et al. 2023 credit it jointly to Welleck et al. and Fu et al., so the exact first source is uncertain.
- No paper defining a "longest repeated substring" degeneration metric was found.

## Q3. What do Xie et al. 2025 and Duan et al. 2026 actually do?

### Takeaway
Both papers exist and are correctly identified. Xie et al. (WSC) is an online chunk-level detector on the final layer that flags repetition once it has started and truncates; Duan et al. is a mechanism-plus-benchmark paper whose early warning combines a last-layer sentence-level linear score with CUSUM and claims a lead of about 1500 tokens, with a high false-positive rate.

### Cited Findings
Xie et al. 2025 (WSC)
- Authors Wenya Xie, Shaochen (Henry) Zhong, Hoang Anh Duy Le, Zhaozhuo Xu, Jianwen Xie, Zirui Liu; EMNLP 2025 main; arXiv 2511.00536; code at GitHub wenyaxie023/WordSaladChopper — [arXiv HTML](https://arxiv.org/html/2511.00536v1); [Stevens record](https://researchwith.stevens.edu/en/publications/word-salad-chopper-reasoning-models-waste-a-ton-of-decoding-budge/)
- Chunks are split on "\n\n"; the feature is the final transformer block output at the chunk-ending "\n\n"; no intermediate layers are used — [arXiv HTML](https://arxiv.org/html/2511.00536v1)
- Labels: a chunk is word salad if its all-MiniLM-L6-v2 embedding has cosine >= 0.99 with any of the previous 100 chunks; every chunk from the earliest chopping point onward is labelled positive, earlier chunks negative; 1,000 s1K traces; classes balanced 1:1 — [arXiv HTML](https://arxiv.org/html/2511.00536v1)
- Classifier: single-layer logistic classifier (BCE, Adam lr 1e-2, 50 epochs). Chop after two consecutive positives, append "I can find a clearer solution if I focus on the core problem.", continue with a 4k (or 8k) budget; initial cap 32k — [arXiv HTML](https://arxiv.org/html/2511.00536v1)
- Over 55% of tokens on GPQA-Diamond are word salad for R1-Distill models, reduced to about 4-6% after chopping; Qwen-1.5B on GSM8K drops from 1,904 to 1,082 tokens with slightly higher accuracy — [arXiv HTML](https://arxiv.org/html/2511.00536v1)
- Framed as detection during generation, not prediction before repetition — [arXiv HTML](https://arxiv.org/html/2511.00536v1)

Duan et al. 2026 (Circular Reasoning)
- Authors Zenghao Duan, Liang Pang, Zihao Wei, Wenbin Duan, Yuxin Tian, Shicheng Xu, Jingcheng Deng, Zhiyi Yin, Xueqi Cheng (ICT, CAS, and People's Public Security University of China); arXiv 2601.05693v1, 9 Jan 2026; no venue listed — [arXiv HTML](https://arxiv.org/html/2601.05693v1)
- LoopBench: 700 samples, 7 subtasks x 100, generated with GPT-5; arithmetic (square root, long division, Newton iteration) and recursive reasoning (truth-teller puzzles, logical paradox, Tower of Hanoi, path planning) — [arXiv HTML](https://arxiv.org/html/2601.05693v1)
- Mechanism claims: "reasoning impasse" triggers the loop; state collapse with entropy drop and logit surge ("determinism surge"); a "V-shaped" attention pattern on attention sinks and the most recent ~128 tokens sustains it; reasoning-graph clusters (K = 200 over sentence-mean final-layer states) become periodic before the text repeats ("semantic attractor"). Analyses are observational on two model families — [arXiv HTML](https://arxiv.org/html/2601.05693v1)
- Early prediction: sentence-mean last-layer states, linear score, CUSUM with reference r = mean of non-repetitive cases and threshold h = alpha * S_max calibrated on 50 non-loop instances per model, persistence p; at least 50 loop and 50 non-loop test cases per model; EDR 0.64-0.76, FPR 0.24-0.34; average lead over 40 sentences (about 1500 tokens) — [arXiv HTML](https://arxiv.org/html/2601.05693v1)
- Other works already build on it: a July 2026 CUSUM monitor for quantized small reasoning models notes that "semantic repetition can precede overt lexical loops" and credits Duan et al. (2026) — [arXiv 2607.20129](https://arxiv.org/pdf/2607.20129)

### Inferences
- Xie et al. are not an early-warning method, since their positives start at the first detected repetition, so contrasting the project's pre-onset warning with WSC is fair.
- Duan et al. do claim pre-onset warning, so the paper should not say prior probes only detect ongoing loops. A fair contrast is that Duan's warning is sentence level on reasoning models, uses greedy decoding on a synthetic loop-inducing benchmark, has FPR around 0.24-0.34 at the operating point, and reads only the last layer.

### Gaps
- Duan's onset definition for the "Repetition Onset" window and the generation cap were in truncated appendices and are not confirmed.

## Q4. Are there datasets or benchmarks of degenerate generations with onset annotations?

### Takeaway
I found no public dataset with human or LLM-judge annotated loop onsets. Existing resources either label whole responses, use automatic rule-based onsets, or provide loop-inducing prompts.

### Cited Findings
- LoopBench (Duan et al. 2026): 700 loop-prone prompts with automatic loop criteria (NLR/SLR); the onset window is described but its annotation procedure is not documented in the main text — [arXiv HTML](https://arxiv.org/html/2601.05693v1)
- A second, unrelated "LoopBench" with "explicit loop-inducing conditions" is introduced by LoopGuard (Apr 2026), and a third "LoopBench" (arXiv 2512.13713) concerns LLM swarms, so the name is ambiguous — [LoopGuard arXiv](https://arxiv.org/abs/2604.10044); [arXiv API listing of 2512.13713](https://arxiv.org/abs/2512.13713)
- Hiraoka and Inui define onset automatically as the first recurrence of a 10-token sequence repeated three times at equal intervals within 100 tokens — [arXiv HTML](https://arxiv.org/html/2410.13497v2)
- WSC derives an automatic chopping point from MiniLM similarity (cosine >= 0.99) — [arXiv HTML](https://arxiv.org/html/2511.00536v1)
- RADAR labels whole trajectories into four states using rules, LLM verification and human review, then lets every prefix inherit the trajectory label; it reports mean loop-onset positions per attack — [arXiv HTML](https://arxiv.org/html/2609.38817v1)
- RecurrentGenerator/RecurrentDetector data: 3,400 prompts (ShareGPT benign plus 200 recurrent prompts per model), response-level labels — [arXiv HTML](https://arxiv.org/html/2503.00416v1)
- CodeRepetEval for structural repetition in code generation (Dong et al., ACL 2025 main) — [arXiv 2505.10402](https://arxiv.org/abs/2505.10402)
- FOCUS & RePAIR (Lee et al., ICML 2026 spotlight) builds "onset-centered positive/negative continuation pairs" for pruned LLMs and splits degeneration into loop-entry risk and loop persistence; the abstract does not say how onsets are located — [arXiv 2608.26676](https://arxiv.org/abs/2608.26676)

### Inferences
- A released set of naturally occurring (non-adversarial) loops from instruct models with LLM-judge onsets across math, code, instruction following and medical prompts would be a new resource.

### Gaps
- Whether FOCUS & RePAIR or LoopGuard release onset-labelled data was not checked beyond abstracts.

## Q5. Repetition in modern instruction-tuned and reasoning models (2024-2026)

### Takeaway
Looping is a documented, current failure in R1-style and Qwen3 models under greedy or low-temperature decoding, in distilled students, in quantized and pruned models, and under adversarial prompts. Model providers address it with sampling and penalty recommendations, not by a fix in the model.

### Cited Findings
- DeepSeek-R1 usage notes recommend temperature 0.5-0.7 (0.6) "to prevent endless repetitions or incoherent outputs" (seen in mirrors of the official README, not the live GitHub page) — [Together AI guide](https://docs.together.ai/docs/prompting-deepseek-r1); [README mirror](https://gitserver.onethingai.com/ai-models/DeepSeek-R1-Distill-Qwen-1.5B/commit/39023d17b2e714350c4f03c17df6c72d33428a10); DeepSeek-R1 report: [arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
- Qwen3 guidance: "DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions", and presence_penalty 0-2 to reduce repetitions, with possible language mixing at high values — [Qwen docs](https://qwen.readthedocs.io/en/latest/_sources/getting_started/quickstart.md.txt)
- **Wait, Wait, Wait... Why Do Reasoning Models Loop?** (Pipis, Garg, Kontonis, Shrivastava, Krishnamurthy, Papailiopoulos; arXiv 2512.12895, ICML 2026 spotlight per v2) | looping at low temperature comes from errors in learning: risk aversion when progress steps are hard to learn and a cyclic step is easy, and temporally correlated errors; larger models loop less, distilled students loop much more than teachers; temperature is "a stopgap" | analysis on a synthetic graph task plus open reasoning models — [arXiv](https://arxiv.org/abs/2512.12895)
- **Circular Reasoning** (Duan et al. 2026) reports loop rates across open reasoning, base, instruct and API models (Gemini-2.5-Pro, Claude Opus 4.1, DeepSeek, Qwen-Plus and others) — [arXiv HTML](https://arxiv.org/html/2601.05693v1)
- **Do NOT Think That Much for 2+3=?** (Chen et al., arXiv 2412.21187) introduces "overthinking" in o1-like models, with redundant reasoning on trivial tasks — [arXiv](https://arxiv.org/abs/2412.21187)
- **Missing Premise exacerbates Overthinking** (Fan, Li, Sun; arXiv 2504.06514) — [arXiv](https://arxiv.org/abs/2504.06514)
- **Stop Overthinking** survey of efficient reasoning (Sui et al., arXiv 2503.16419) — [arXiv](https://arxiv.org/abs/2503.16419)
- **Extreme Low-Bit Inference in Reasoning Models** (Alimaskina et al., arXiv 2606.02011): 2-bit Qwen3 reasoning models produce repetitive loops and budget exhaustion; a "loop rescue" detector plus FP16 fallback raises Qwen3-8B MATH-500 accuracy from 17.2% to 74.2% — [arXiv](https://arxiv.org/abs/2606.02011)
- **FOCUS & RePAIR** (arXiv 2608.26676, ICML 2026): pruning amplifies repetition loops even when perplexity and task accuracy are almost unchanged — [arXiv](https://arxiv.org/abs/2608.26676)
- **From Loops to Oops** (Ivgi, Yoran, Berant, Geva; arXiv 2407.06071, NeurIPS 2024): repetition is a fallback under epistemic uncertainty; stronger models shift from repetition to degenerate text to hallucination, and within one generation the order reverses as uncertainty grows; sampling cuts repetition but increases hallucinations — [arXiv](https://arxiv.org/abs/2407.06071v2); [NeurIPS](https://neurips.cc/virtual/2024/105336)
- **Solving LLM Repetition Problem in Production** (Wang, Zou, Min; arXiv 2512.04419): three repetition patterns in batch code-interpretation tasks; fixes are beam search with early_stopping, presence_penalty, and DPO — [arXiv](https://arxiv.org/abs/2512.04419)
- Code generation: structural repetition and RPG grammar-based penalisation (Dong et al., ACL 2025) — [arXiv 2505.10402](https://arxiv.org/abs/2505.10402); repetition in 19 code LLMs and the rule-based DeRep (Liu et al. 2025) — [arXiv 2504.12608](https://arxiv.org/abs/2504.12608)
- Machine translation with LLMs: repetition and language mismatch reduced by model editing (Wang, Li, Lian; EMNLP 2024 main) — [arXiv 2410.07054](https://arxiv.org/abs/2410.07054)
- Adversarial induction of loops or non-termination: Non-Halting Queries, which prove that at temperature 0 a cycle observed beyond the context size never halts and send aligned models including GPT-4o into non-halting states (arXiv 2410.06287) — [arXiv](https://arxiv.org/abs/2410.06287); Engorgio prompts (ICLR 2025, 13 open LLMs) — [arXiv 2412.19394](https://arxiv.org/abs/2412.19394); LoopLLM, which induces low-entropy repetition loops reaching over 90% of the output limit (AAAI 2026) — [arXiv 2511.07876](https://arxiv.org/abs/2511.07876); RECUR reflection-exhaustion attack (arXiv 2602.08214) — [arXiv](https://arxiv.org/abs/2602.08214); Copy-Guided Attacks that induce infinite loops in reasoning LLMs (arXiv 2507.16773) — [arXiv](https://arxiv.org/abs/2507.16773)
- Survey of resource-consumption threats that groups these looping, overthinking and non-termination works (Zhang et al., arXiv 2603.16068v3) — [arXiv HTML](https://arxiv.org/html/2603.16068v3)

### Inferences
- Most 2025-2026 work studies reasoning models or attack-induced loops. Natural loops in standard instruct models on ordinary prompts, at the recommended sampling settings and a 4096-token cap, are less studied, which supports the project's choice of Apertus, Llama-3.1 and Mistral instruct models.

### Gaps
- I did not find a systematic measurement of natural loop rates for Llama-3.1-8B-Instruct or Mistral-7B-Instruct at default sampling, apart from Duan's greedy loop-rate table.

## Q6. Causes and mechanisms of repetition

### Takeaway
The main accounts are self-reinforcement of repeated content (probability amplification), copying circuits (induction heads and downstream repetition neurons), attention-sink interactions, training-data repetition, and learning errors that make cyclic steps easier than progress steps. Most evidence is observational or comes from small or older models.

### Cited Findings
- Positive feedback loop of repeated phrases (GPT-2 Large) — [Holtzman et al. 2020](https://ar5iv.labs.arxiv.org/html/1904.09751)
- Self-reinforcement: the more a sentence has been repeated, the more likely it is to be repeated again, stronger for sentences with high initial probability (16-layer Wikitext-103 transformer, GPT-2, BART) — [Xu et al. 2022, DITTO](https://arxiv.org/abs/2206.02369)
- High-inflow words and Markov upper bounds on repetition probability — [Fu et al. 2021](https://arxiv.org/pdf/2012.14660)
- Training-data repetition is a common factor behind earlier explanations; model size alone does not fix it (OPT-66B still has high rep-2); Alpaca instruction tuning of Llama2-7B reduces repetition — [Li et al. 2023](https://arxiv.org/html/2310.10226v1)
- Learning repeated data hurts copying/induction circuits (relevant background) — Hernandez et al. 2022, [arXiv 2205.10487](https://arxiv.org/abs/2205.10487)
- Induction heads as the in-context copying mechanism — Olsson et al. 2022, [arXiv 2209.11895](https://arxiv.org/abs/2209.11895)
- "Induction head toxicity": induction heads dominate output logits during repetition and exclude other heads; proposes attention-head regularisation. arXiv comment says "Need to be refined" — Wang, Yin, Leong, Zhang, et al., [arXiv 2505.13514](https://arxiv.org/abs/2505.13514)
- Repetition neurons, middle-layer detectors and final-layer replicators — [Hiraoka and Inui 2025](https://arxiv.org/html/2410.13497v2); their effect on ICL depends on layer depth, compared with induction heads — Doan, Hiraoka, Inui, [arXiv 2507.07810](https://arxiv.org/abs/2507.07810)
- Distinct mechanisms sustain repetition in natural versus ICL settings (strong single heads plus diffuse contributions in natural repetition; non-cyclical heads in ICL) — Mahaut and Franzon, [arXiv 2504.01100](https://arxiv.org/pdf/2504.01100)
- Repeated-token divergence is tied to attention sinks: the first-layer mechanism that marks the first token also marks repeated identical tokens (ICML 2025) — Yona et al., [arXiv 2503.08908](https://arxiv.org/abs/2503.08908); [ICML](https://icml.cc/virtual/2025/poster/45013); attention sinks background — Xiao et al., [arXiv 2309.17453](https://arxiv.org/abs/2309.17453)
- V-shaped attention on sinks and recent tokens, entropy collapse and periodic semantic states before textual repetition in reasoning models — [Duan et al. 2026](https://arxiv.org/html/2601.05693v1)
- Attention heads locking onto a narrow history suffix — [LoopGuard](https://arxiv.org/abs/2604.10044); prompt-attention share falling and cross-context similarity changing before loop onset — [RADAR](https://arxiv.org/html/2609.38817v1)
- Learning errors (risk aversion, temporally correlated errors) and distillation as causes in reasoning models — [Pipis et al.](https://arxiv.org/abs/2512.12895)
- Loop entry risk versus persistence, with persistence governed by "escape mass" on plausible alternatives — [FOCUS & RePAIR](https://arxiv.org/abs/2608.26676)
- Repetition as a fallback under epistemic uncertainty — [Ivgi et al. 2024](https://arxiv.org/abs/2407.06071v2)
- Attractor-like dynamics from a Markov view of self-attention (Ildiz et al. 2024) [UNVERIFIED id], and MoE expert-routing imbalance causing repetition under stress (RepetitionCurse, Huang et al. 2025d) [UNVERIFIED] — both cited in [survey 2603.16068](https://arxiv.org/html/2603.16068v3)

### Inferences
- Mechanism papers mostly characterise the loop once it is running (neurons found by after-minus-before onset differences, logit dominance during repetition). Duan, RADAR and Pipis are the only works that describe a pre-loop state (impasse, attention drift, cyclic step preferred), which is the regime an early-warning probe targets.

### Gaps
- No causal study links a specific pre-onset internal direction to later loop entry in standard instruct models.

## Q7. Mitigation at training and decoding time

### Takeaway
Mitigations split into training objectives (unlikelihood, ScaleGrad, DITTO, self-contrastive, Rep-Dropout, DPO, SAE-guided RL), decoding rules (repetition, presence and frequency penalties, contrastive search, contrastive decoding, truncation and typical sampling, penalty decoding, grammar penalties), and reactive loop breakers in reasoning models (chunk chopping, early exit, KV pruning, attention realignment, fallback models). Few of them use a learned internal-state trigger.

### Cited Findings
Training time
- Token- and sequence-level unlikelihood training — [Welleck et al. 2020](https://ar5iv.labs.arxiv.org/html/1908.04319)
- ScaleGrad, "Straight to the Gradient" (Lin, Han, Joty; ICML 2021) — [arXiv 2106.07207](https://arxiv.org/abs/2106.07207)
- DITTO pseudo-repetition penalisation — [arXiv 2206.02369](https://arxiv.org/abs/2206.02369)
- Self-contrastive training (Guan and Huang; ACL 2023 Findings) — [arXiv 2307.01542](https://arxiv.org/abs/2307.01542)
- Repetition Dropout — [Li et al. 2023](https://arxiv.org/abs/2310.10226)
- Attention-head regularisation against induction-head dominance — [arXiv 2505.13514](https://arxiv.org/abs/2505.13514)
- Rebalanced encoding against high inflow — [Fu et al. 2021](https://arxiv.org/pdf/2012.14660)
- SAE-derived signals in SFT/RL to reduce repetition — [Qwen-Scope](https://arxiv.org/abs/2605.11887)
- DPO in production — [arXiv 2512.04419](https://arxiv.org/abs/2512.04419)
- FOCUS distillation reweighting and RePAIR onset-centred margin loss for pruned models — [arXiv 2608.26676](https://arxiv.org/abs/2608.26676)
- HAPAX, excluding induction-head-predictable tokens from the loss (Sahin et al. 2025) [UNVERIFIED] — cited in [survey 2603.16068](https://arxiv.org/html/2603.16068v3)

Decoding time
- Repetition penalty (CTRL, Keskar et al. 2019) — [arXiv 1909.05858](https://arxiv.org/abs/1909.05858); presence and frequency penalties are cited jointly with Keskar et al. in [WSC](https://arxiv.org/html/2511.00536v1) and recommended by [Qwen docs](https://qwen.readthedocs.io/en/latest/_sources/getting_started/quickstart.md.txt)
- Nucleus sampling — [Holtzman et al. 2020](https://arxiv.org/abs/1904.09751); Mirostat (Basu et al., ICLR 2021) — [arXiv 2007.14966](https://arxiv.org/abs/2007.14966); Locally Typical Sampling (Meister et al., TACL 2022) — [arXiv 2202.00666](https://arxiv.org/abs/2202.00666); "Closing the Curious Case of Neural Text Degeneration" (Finlayson, Hewitt, Koller) — [arXiv 2310.01693](https://arxiv.org/abs/2310.01693)
- Contrastive search / SimCTG (Su et al., NeurIPS 2022) — [arXiv 2202.06417](https://arxiv.org/abs/2202.06417); Contrastive Decoding (Li et al.) — [arXiv 2210.15097](https://arxiv.org/abs/2210.15097)
- Penalty decoding against self-reinforcement (Zhu, Hao, Wang 2023) — [arXiv 2310.14971](https://arxiv.org/abs/2310.14971)
- RPG grammar-based penalisation for code — [arXiv 2505.10402](https://arxiv.org/abs/2505.10402); DeRep rule-based repair — [arXiv 2504.12608](https://arxiv.org/abs/2504.12608)
- Beam search with early_stopping as a production fix — [arXiv 2512.04419](https://arxiv.org/abs/2512.04419)
- Targeted patch for the repeated-token attention-sink failure — [Yona et al. 2025](https://arxiv.org/abs/2503.08908)
- Neuron deactivation after onset — [Hiraoka and Inui](https://arxiv.org/html/2410.13497v2); SAE feature deactivation — [Yao et al.](https://arxiv.org/abs/2504.14218v3)

Reasoning-model loop breaking and early stopping
- WSC chop-and-regenerate — [arXiv 2511.00536](https://arxiv.org/html/2511.00536v1)
- LoopGuard KV-cache pruning — [arXiv 2604.10044](https://arxiv.org/abs/2604.10044)
- RADAR attention realignment: loop rate 37.0% to 27.2% macro average, benign accuracy 67.5% to 71.5%, only tested under attacks — [arXiv HTML](https://arxiv.org/html/2609.38817v1)
- DTS decoding-tree sketching, favouring shorter completed trajectories — [arXiv 2511.00640](https://arxiv.org/abs/2511.00640)
- Loop rescue with FP16 fallback for 2-bit models — [arXiv 2606.02011](https://arxiv.org/abs/2606.02011)
- CUSUM-shaped monitoring with targeted re-decoding for quantized small reasoning models (uncertainty, repetition and local-change features; MATH-500 54.73% to 56.40%, no gain on Omni-MATH) — [arXiv 2607.20129](https://arxiv.org/abs/2607.20129)
- Early exit by answer confidence: DEER (arXiv 2504.15895) — [arXiv](https://arxiv.org/abs/2504.15895); Dynasor / Certaindex (arXiv 2412.20993) — [arXiv](https://arxiv.org/abs/2412.20993); hidden-state correctness probes — [arXiv 2504.05419](https://arxiv.org/abs/2504.05419); ESC low-entropy window stopping (Li et al. 2024), as described in [WSC](https://arxiv.org/html/2511.00536v1)
- Suppression of self-affirmation reflections — [arXiv 2506.12353](https://arxiv.org/abs/2506.12353)

### Inferences
- Nearly all decoding mitigations are applied to every token whether or not a loop is coming, which is where they cost quality (Qwen docs note language mixing at high presence_penalty; Ivgi et al. note sampling trades repetition for hallucination). A calibrated early warning could gate these interventions, which is a natural motivation for the project.

### Gaps
- No head-to-head comparison of an internal-state trigger against always-on penalties on the same instruct models was found.
