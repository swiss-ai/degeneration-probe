# Reference audit for the early-warning probe workshop draft

Scope: all 19 entries in `paper/references.bib` and all 41 `\cite`/`\citep`/`\citet` uses in `paper/main.tex` (draft as of 2026-10-12). "Verified" means I read the arXiv abstract page, arXiv HTML full text, ACL Anthology, PMLR or ICML page during this session. "From knowledge" means I did not re-check it in this session.

Inventory:
- Cited in main.tex (15 keys): holtzman2020curious, welleck2020neural, pipis2025wait, xie2025wordsalad, yu2025breaking, duan2026circular, apertus2025, grattafiori2024llama, li2025fineharm, niu2024ragtruth, qi2025shallow, orgad2025llms, mckenzie2025highstakes, hedstrom2025mera, hu2022lora.
- In the bib but never cited (4 keys): manber1993suffix, zheng2025processbench, han2024wildguard, baker2025monitoring. natbib drops them from the printed bibliography, so this is harmless, but manber1993suffix is the natural citation for the LRS baseline (suffix-array longest repeated substring) and is not used there.
- Cited keys missing from the bib: none.
- The draft studies two models (Apertus-8B-Instruct, Llama-3.1-8B-Instruct). Mistral is never mentioned in main.tex and has no bib entry. If Mistral results are added, a Mistral 7B citation (Jiang et al. 2023, arXiv:2310.06825, from knowledge) is needed.

## 1. Is each reference real and correctly described in the bib?

### Takeaway
All 19 references are real. The cited 15 have correct titles and author lists. Three entries carry an outdated venue (Pipis is now ICML 2026, Li/FineHarm is NeurIPS 2025, and the Hedström entry lacks pages and arXiv id); the rest need only cosmetic additions (DOIs, eprint fields).

### Cited Findings
- holtzman2020curious: title, authors (Holtzman, Buys, Du, Forbes, Choi) and ICLR 2020 are correct (from knowledge; content verified via ar5iv) — [ar5iv 1904.09751](https://ar5iv.labs.arxiv.org/html/1904.09751)
- welleck2020neural: title and authors (Welleck, Kulikov, Roller, Dinan, Cho, Weston) verified; arXiv:1908.04319; ICLR 2020 from knowledge — [arXiv 1908.04319](https://arxiv.org/abs/1908.04319)
- orgad2025llms: title and all 7 authors verified on arXiv (2410.02707, v4 18 May 2025); ICLR 2025 confirmed by ML Anthology listing; the OpenReview id KRnsX5Em3W could not be opened (OpenReview served a browser check) — [arXiv 2410.02707](https://arxiv.org/abs/2410.02707); [ML Anthology](https://mlanthology.org/authors/k/kotek-hadas/)
- mckenzie2025highstakes: title and 7 authors verified; arXiv comment says "Accepted at NeurIPS 2025"; latest version v4, 23 Jan 2026 — [arXiv 2506.10805](https://arxiv.org/abs/2506.10805)
- hedstrom2025mera: title, 5 authors, ICML 2025, PMLR vol. 267 verified; pages 22924–22945; arXiv:2510.13290 — [PMLR](https://proceedings.mlr.press/v267/hedstrom25a.html); [arXiv 2510.13290](https://arxiv.org/html/2510.13290v1)
- duan2026circular: title, 9 authors in the bib order, arXiv:2601.05693 (v1, 9 Jan 2026) verified; no venue listed on arXiv — [arXiv 2601.05693](https://arxiv.org/abs/2601.05693)
- xie2025wordsalad: EMNLP 2025 main, pages 33588–33598, DOI 10.18653/v1/2025.emnlp-main.1705, 6 authors verified; arXiv:2511.00536. Official title capitalisation is "Reasoning Models Waste A Ton Of Decoding Budget On Useless Repetitions, Self-Knowingly" — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1705/); [arXiv 2511.00536](https://arxiv.org/abs/2511.00536)
- yu2025breaking: title and 5 authors (Yu, Liu, Sun, Shi, Chen) verified; arXiv:2503.00416 (1 Mar 2025, cs.CR); no venue found — [arXiv 2503.00416](https://arxiv.org/abs/2503.00416)
- pipis2025wait: title and 6 authors verified; arXiv comment "Published as a Spotlight at ICML 2026", v2 revised 24 Aug 2026; ICML 2026 listing exists — [arXiv 2512.12895](https://arxiv.org/abs/2512.12895); [ICML 2026 spotlight](https://icml.cc/virtual/2026/spotlight/84640)
- apertus2025: title verified; the arXiv byline starts with the collective name "Project Apertus" followed by 102 named authors, so `{{Project Apertus}}` is a reasonable author field — [arXiv 2509.14233](https://arxiv.org/abs/2509.14233)
- grattafiori2024llama: author order Grattafiori, Dubey, Jauhri, Pandey, Kadian, ... verified (v3, 23 Nov 2024) — [arXiv 2407.21783](https://arxiv.org/abs/2407.21783)
- hu2022lora: 8 authors in the bib order verified; ICLR 2022 from knowledge — [arXiv 2106.09685](https://arxiv.org/abs/2106.09685)
- li2025fineharm: title and 5 authors verified; arXiv comment "NeurIPS 2025 Accepted Paper" (v3, 22 Sep 2025). The bib still lists it as an arXiv preprint — [arXiv 2506.09996](https://arxiv.org/abs/2506.09996)
- niu2024ragtruth: title, 8 authors, ACL 2024 Vol. 1, pages 10862–10878 verified; DOI 10.18653/v1/2024.acl-long.585 — [ACL Anthology](https://aclanthology.org/2024.acl-long.585/)
- qi2025shallow: 8 authors verified; ICLR 2025, one of the ICLR 2025 Outstanding Paper winners — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html); [ICLR blog](https://blog.iclr.cc/2025/04/22/announcing-the-outstanding-paper-awards-at-iclr-2025/)
- Uncited entries manber1993suffix (SIAM J. Comput. 22(5):935–948, 1993), zheng2025processbench (ACL 2025), han2024wildguard (NeurIPS 2024 D&B), baker2025monitoring (arXiv:2503.11926): metadata matches what I know but I did not re-verify them in this session, since they are not cited.

### Corrected BibTeX (only entries that should change)

```bibtex
@inproceedings{pipis2025wait,
  title     = {Wait, Wait, Wait... Why Do Reasoning Models Loop?},
  author    = {Pipis, Charilaos and Garg, Shivam and Kontonis, Vasilis and
               Shrivastava, Vaishnavi and Krishnamurthy, Akshay and
               Papailiopoulos, Dimitris},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  note      = {Spotlight. arXiv:2512.12895}
}

@inproceedings{li2025fineharm,
  title     = {From Judgment to Interference: Early Stopping {LLM} Harmful Outputs
               via Streaming Content Monitoring},
  author    = {Li, Yang and Sheng, Qiang and Yang, Yehan and Zhang, Xueyao and Cao, Juan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  note      = {arXiv:2506.09996}
}

@inproceedings{hedstrom2025mera,
  title     = {To Steer or Not to Steer? {M}echanistic Error Reduction with Abstention
               for Language Models},
  author    = {Hedstr{\"o}m, Anna and Amoukou, Salim I. and Bewley, Tom and
               Mishra, Saumitra and Veloso, Manuela},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {22924--22945},
  publisher = {PMLR},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/hedstrom25a.html},
  note      = {arXiv:2510.13290}
}

@inproceedings{xie2025wordsalad,
  title     = {Word Salad Chopper: Reasoning Models Waste A Ton Of Decoding Budget
               On Useless Repetitions, Self-Knowingly},
  author    = {Xie, Wenya and Zhong, Shaochen and Le, Hoang Anh Duy and Xu, Zhaozhuo
               and Xie, Jianwen and Liu, Zirui},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural
               Language Processing},
  pages     = {33588--33598},
  address   = {Suzhou, China},
  publisher = {Association for Computational Linguistics},
  year      = {2025},
  doi       = {10.18653/v1/2025.emnlp-main.1705}
}

@inproceedings{niu2024ragtruth,
  title     = {{RAGT}ruth: A Hallucination Corpus for Developing Trustworthy
               Retrieval-Augmented Language Models},
  author    = {Niu, Cheng and Wu, Yuanhao and Zhu, Juno and Xu, Siliang and
               Shum, KaShun and Zhong, Randy and Song, Juntong and Zhang, Tong},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for
               Computational Linguistics (Volume 1: Long Papers)},
  pages     = {10862--10878},
  year      = {2024},
  doi       = {10.18653/v1/2024.acl-long.585}
}
```

Optional additions: `note = {arXiv:2410.02707}` for orgad2025llms; `note = {arXiv:1908.04319}` for welleck2020neural; `note = {arXiv:1904.09751}` for holtzman2020curious; `note = {arXiv:2106.09685}` for hu2022lora; `url` to the ICLR proceedings page for qi2025shallow.

### Inferences
- The proceedings number "43rd" for ICML 2026 follows from ICML 2025 being the 42nd (PMLR page above); I did not see the ICML 2026 PMLR volume.
- None of the references look fabricated. The two 2026 items (duan2026circular, pipis2025wait v2) exist with the exact titles and authors in the bib.

### Gaps
- OpenReview pages (Orgad KRnsX5Em3W, Hedström fUCPq5RvmH) could not be loaded because of a browser check, so the OpenReview ids are unconfirmed. papers.cool lists fUCPq5RvmH as an OpenReview id for the Hedström paper ([search result](https://papers.cool/venue/fUCPq5RvmH@OpenReview)), which supports it.
- No ICML 2026 PMLR volume or page numbers found for Pipis et al.

## 2. Is each citing sentence accurate to the cited work?

### Takeaway
Most attributions are accurate and several numbers check out exactly. Five need fixing: (i) Hedström/MERA is cited for "adapting the underlying representations", but MERA is inference-time activation steering with no weight or adapter training; (ii) Pipis et al. never discuss entropy, so they do not support the windowed-entropy baseline, and Holtzman et al. support it only indirectly; (iii) Duan et al.'s persistence ablation raises FPR to 0.40–0.62, above 50% in only 4 of 8 models; (iv) Duan's 0.991–0.998 accuracy range mixes numeral loops (single token vectors) with statement loops (sentence means), and the layer for that table is not stated; (v) the rep/ℓ baseline comes from Welleck et al. but is not cited there. There are also citation-command errors (`\citet` vs `\citep`).

### Cited Findings

**Intro, line 142–145: "LMs can enter repetitive loops ... observed in both neural text generation and modern reasoning models [holtzman, welleck, pipis]."** Accurate.
- Holtzman: "The probability of a repeated phrase increases with each repetition, creating a positive feedback loop" (Fig. 4 caption) — [ar5iv 1904.09751](https://ar5iv.labs.arxiv.org/html/1904.09751)
- Pipis studies reasoning models that "often loop, repeating the same text at low temperatures or with greedy decoding" — [arXiv 2512.12895](https://arxiv.org/abs/2512.12895)

**Lines 146 and 164: "repetitive degeneration can be detected from internal representations"; "Prior activation-based detectors report strong classification performance [xie, yu]."** Accurate.
- WSC: hidden states at the "\n\n" token after each chunk feed "a single-layer logistic classifier"; Qwen-7B τ=0 accuracy/AUROC 89.77–93.52 / 95.84–98.63 — [arXiv HTML 2511.00536](https://arxiv.org/html/2511.00536v1)
- Yu: RecurrentDetector reports 95.24% accuracy and F1 0.87 — [arXiv 2503.00416](https://arxiv.org/abs/2503.00416)

**Line 148: "[duan] further show that sentence-level scores from last-layer hidden-state can be accumulated with CUSUM to predict statement loops before explicit textual repetition begins."** Accurate. Duan: "for each sentence i, we take the average of its hidden states at the last layer"; CUSUM S_i = max(0, S_{i-1} + (x_i − r)); claim that "semantic circularity consistently precedes explicit textual repetition" (support is qualitative case studies) — [arXiv HTML 2601.05693](https://arxiv.org/html/2601.05693v1). Formatting: this is `\citep{duan2026circular}` used as a sentence subject; it should be `\citet`. Also "last-layer hidden-state" should be "hidden states".

**Line 176–177: "unlike prior work evaluated at low temperature or on balanced test sets [xie, duan]."** Mostly accurate.
- Duan prediction: "using greedy decoding to ensure reproducibility" and "a balanced test set for each model, comprising at least 50 repetitive and 50 non-repetitive cases" — [arXiv HTML 2601.05693](https://arxiv.org/html/2601.05693v1)
- WSC: "unless otherwise specified, all reported models are of DeepSeek-R1-Distill series with temp τ=0"; several tables (4, 8, 12) use τ=0.6; the training set is rebalanced 1:1, test balance is not stated — [arXiv HTML 2511.00536](https://arxiv.org/html/2511.00536v1)
- Yu could be added: generation experiments use temperature 0 with max_new_tokens 2000 — [arXiv HTML 2503.00416](https://arxiv.org/html/2503.00416)

**Line 250–254 (Preliminaries): "[xie] report all results at temperature τ=0 unless stated otherwise, and [duan] evaluate their early-warning method on greedily decoded generations ..., although their benchmark-level loop statistics use a balanced sampling configuration."** Accurate in substance.
- WSC's wording is about models, not results ("all reported models ... with temp τ=0" unless specified) — [arXiv HTML 2511.00536](https://arxiv.org/html/2511.00536v1)
- Duan's "Balanced" decoding setting is a named configuration (T=0.6, top-k 15, top-p 0.9, with a fixed repetition penalty), next to "Conservative" (T=0.1) and "Explorative" (T=1.0); table captions are inconsistent about which setting each table uses — [arXiv HTML 2601.05693](https://arxiv.org/html/2601.05693v1)
- Formatting: both cites are `\citep` used as subjects; should be `\citet`. Writing "a balanced sampling configuration" right after "balanced test sets" invites confusion; naming the setting ("their 'Balanced' decoding setting, T=0.6") avoids it.

**Line 255–257: "low-temperature and likelihood-maximising decoding is known to make repetitive generation substantially more prevalent [holtzman]."** Accurate. Holtzman Table 1 repetition %: greedy 73.66, beam (b=16) 28.94, top-k k=40 0.78 vs top-k k=40 with t=0.7 8.86, pure sampling 0.22, human 0.28 — [ar5iv 1904.09751](https://ar5iv.labs.arxiv.org/html/1904.09751)

**Line 297: "which token positions are selected \citet{orgad2025llms, mckenzie2025highstakes}"**. Partly accurate.
- Orgad: "truthfulness information is concentrated in the exact answer tokens", compared against last generated token, the token before it, and the final prompt token — [arXiv HTML 2410.02707](https://arxiv.org/html/2410.02707v4)
- McKenzie compares aggregation architectures over positions (mean, max, last-token, softmax, attention, max of rolling means) with one label per sample; "The Last Token activation probe consistently performs poorly"; the paper does not discuss per-token supervision — [arXiv HTML 2506.10805](https://arxiv.org/html/2506.10805v4)
- So Orgad supports "position choice matters"; McKenzie supports "how per-position scores are aggregated matters". Neither proposes the draft's selection schemes (rollout_balanced, random_window, frontier_window). "Motivated by" fits better than "follows". Formatting: should be `\citep`.

**Line 297: "whether the underlying representations are adapted \citet{hedstrom2025mera}".** Inaccurate.
- MERA trains a linear error probe per layer, takes the steering vector v* = λ*·w with λ* = max(0, (α − wᵀh)/‖w‖²), and is activation steering at inference; the paper contrasts itself with weight-changing fine-tuning and does not mention LoRA or adapters — [arXiv HTML 2510.13290](https://arxiv.org/html/2510.13290v1)
- The adapted regime is LoRA trained jointly with the probe, which MERA does not do. Fix: cite hu2022lora for the adaptation axis, and keep MERA only for "probe-guided intervention with calibrated abstention" (it fits the related-work paragraph and the alarm-calibration paragraph). Formatting: should be `\citep`.

**Line 307 / 714: LoRA adapters [hu2022lora]; "train a small number of additional parameters while leaving the pretrained weights unchanged".** Accurate (from knowledge of the LoRA paper; authors verified at [arXiv 2106.09685](https://arxiv.org/abs/2106.09685)). "Standard way to modify a model's representations at low cost" is an opinion and fine as such.

**Line 320–321 and 933–934: windowed `seq-rep-2` [welleck].** Accurate as an adaptation. Welleck defines seq-rep-n as "the portion of duplicate n-grams (seq-rep-n) in a generated sequence", computed "on the continuation", and reports n=4 — [ar5iv 1908.04319](https://ar5iv.labs.arxiv.org/html/1908.04319). The draft's formula 1 − |uniq bigrams|/(|W|−1) matches the definition with n=2 applied to a trailing window.

**Rep/ℓ baseline (lines 316–318, 961–988), no citation.** The name and the ℓ=128 choice come from Welleck: "the fraction of next-token (top-1) predictions that occur in the previous ℓ tokens (rep/ℓ)", computed on argmax predictions given ground-truth context; Welleck reports the Transformer's predictions "appeared in the preceding 128 words 62% of the time" — [ar5iv 1908.04319](https://ar5iv.labs.arxiv.org/html/1908.04319). The draft computes it on sampled tokens, not top-1 predictions, so "adapted from rep/ℓ [welleck2020neural]" is the right wording.

**Line 325: windowed entropy, "lower entropy producing a higher degeneration score [holtzman, pipis]".** Weakly supported.
- Pipis never uses the word "entropy"; the closest passage is that the model, "as it repeats the same text, ... becomes increasingly confident" — [arXiv HTML 2512.12895](https://arxiv.org/html/2512.12895v2)
- Holtzman does not discuss next-token entropy directly; it contrasts flat vs peaked next-token distributions (Fig. 5) and shows repeated phrases gain probability — [ar5iv 1904.09751](https://ar5iv.labs.arxiv.org/html/1904.09751)
- Suggested wording: "motivated by the observation that repetition makes the model increasingly confident [holtzman, pipis]". Also a missing space before `\citep` on line 325.

**Line 485 and 394: "Prior work attaches the probe to the final layer [xie, duan]".** Accurate.
- WSC: "we collect the output of the final transformer block of each <\n\n>"; no layer comparison — [arXiv HTML 2511.00536](https://arxiv.org/html/2511.00536v1)
- Duan: the CUSUM predictor uses "the average of its hidden states at the last layer"; layer-wise analysis exists only for cosine/L2 statistics, not classifiers; the layer behind Table 2 is not stated — [arXiv HTML 2601.05693](https://arxiv.org/html/2601.05693v1)
- Yu uses all decoder layers, so "prior work" should stay restricted to these two (the related-work section already says "either to the final layer or to an aggregation over all layers").

**Lines 642–645 (related work): Holtzman and Welleck.** Accurate (see quotes above). Welleck frames repetition as a training-objective problem: likelihood training "is a root cause" of dull, repetitive text — [arXiv 1908.04319](https://arxiv.org/abs/1908.04319).

**Lines 651–655: Pipis "looping is common at low temperature and under greedy decoding, distilled models loop far more than their teachers, ... temperature as a stopgap rather than a fix".** Accurate. Abstract: "temperature is a stopgap rather than a holistic solution"; Section 2: students "loop far more than their teachers" — [arXiv HTML 2512.12895](https://arxiv.org/html/2512.12895v2)

**Lines 659–661: Duan "classify a sentence from the mean of its last-layer hidden states, reporting 0.991–0.998 accuracy and 0.996–1.000 AUC across linear, kernel and non-linear classifiers."** Numbers correct, description imprecise.
- Table 2 (DS-Qwen-14B, Qwen3-8B; Linear, SVM, MLP): ACC 0.991–0.998, AUC 0.996–1.000 across numeral and statement loops combined. Statement loops alone: ACC 0.997–0.998, AUC 0.998–1.000. Duan uses "averaged hidden states for statement loops and individual vectors for numerical loops", and the main text does not name the layer for Table 2 — [arXiv HTML 2601.05693](https://arxiv.org/html/2601.05693v1)
- Fix: "classify statement-loop sentences from mean-pooled hidden states (and numeral-loop tokens from single states), reporting ...", or restrict the range to statement loops.

**Lines 662–669: WSC "classify a chunk ... from the final block's output at the chunk's trailing newline token, reporting 89.8–93.5% accuracy and 95.8–98.6 AUROC at temperature zero and somewhat lower figures at temperature 0.6; ... the model being aware of the pattern; ... every chunk from the earliest chopping point onward is relabelled positive, and an intervention is triggered only after two consecutive positive chunks."** Accurate with two caveats.
- The ranges are for one model (DeepSeek-R1-Distill-Qwen-7B, Table 5); Qwen3-8B reaches only 78.0–88.1 accuracy (Table 12). At τ=0.6 AIME25 drops to 77.96 / 80.15, which is more than "somewhat lower" — [arXiv HTML 2511.00536](https://arxiv.org/html/2511.00536v1)
- "LRMs are self-aware when trapped in these loops"; chunks before the earliest chopping point are labelled 0 and "all chunks including and after" are labelled 1; Figure 1 rule "two consecutive word salad chunks detected", with streak_len=2 for long chunks and short_streak_len=5 for chunks of ≤10 tokens — same source

**Lines 670–676: Yu "small MLP over per-token similarity features ... binary MLP-neuron activation patterns ... across all decoder layers; detection runs online from the 400th generated token on and reaches 95.24% accuracy, 0.87 F1 and a 2.6% false-positive rate, on prompts found by an evolutionary search."** Accurate with one caveat.
- Three linear layers (d → d/2 → d/4 → 1, with LayerNorm and ReLU, d=120); input is per-token maximum similarity to other tokens' binary activation states, sorted and unsorted, concatenated; FPR 0.0259; recall 0.8288 — [arXiv HTML 2503.00416](https://arxiv.org/html/2503.00416)
- The 400-token start is given as an example ("e.g., 400 tokens", the 78th percentile of ShareGPT response lengths), not a fixed setting. Writing "after a minimum length (400 tokens in their example)" is safer — same source
- The test set mixes 2,200 benign and 1,200 recurrent prompts (about 65/35), and the text elsewhere says "4,000 prompts" — same source

**Lines 676–682: "All three classify text that is already inside the repetitive pattern, report aggregate classification quality, and fix the readout"; Duan tracks cosine and L2 by layer.** Accurate. Duan computes "the cosine similarity and L2 norm differences between activation vectors of identical tokens" across cycles, by layer and module — [arXiv HTML 2601.05693](https://arxiv.org/html/2601.05693v1)

**Lines 684–692: Duan "early-detection rates of 0.64–0.76 at false-positive rates of 0.24–0.34, with average lead times of 1306–1872 tokens, and show that dropping the persistence requirement raises the false-positive rate above 50%. ... Their evaluation uses greedy decoding and balanced test sets drawn from a benchmark constructed to induce loops."**
- With persistence (8 models): EDR 0.64–0.76, FPR 0.24–0.34, ATLT 1305.9–1871.6 tokens, ASLT 36.5–51.4 sentences. These match — [arXiv HTML 2601.05693](https://arxiv.org/html/2601.05693v1)
- Without persistence ("w/o p=1"): FPR 0.40–0.62; above 0.50 for 4 of 8 models (0.54, 0.52, 0.62, 0.58); EDR rises to 0.72–0.86. The draft overstates; write "raises the false-positive rate to 0.40–0.62" — same source
- Per-model CUSUM settings: persistence p ∈ {3,4,5}, threshold scale α ∈ {1.2,…,1.5} (Appendix D.2, Table 10) — same source
- LoopBench: seven sub-tasks chosen because "looping behavior are primarily induced by two specific task patterns", 100 GPT-5-synthesised samples each; the system instruction is meant "to trigger the model's maximum reasoning potential". "Constructed to induce loops" is a fair reading, but Duan's own framing is "tasks observed to induce loops"; "built from task types observed to trigger loops" is closer — same source
- Lead time is reported in sentences and tokens; text says "over 40 sentences (or 1500 tokens)" — same source

**Lines 693–696: WSC appendix trajectory "score rises from 1.2×10⁻¹⁰ to 1.000 over 221 chunks, so it is near zero on the first repetitive chunks and certain only much later."** Numbers correct, interpretation slightly beyond the source. Table 10 (MATH-500 #462, Qwen-7B, τ=0.6) lists chunk 209 at 1.19e-10, chunk 255 at 3.69e-5, chunk 430 at 1.000; the paper says later chunks become easier to identify as repetitions accumulate, but does not mark chunk 209 as the first repetitive chunk or give the onset — [arXiv HTML 2511.00536](https://arxiv.org/html/2511.00536v1). Safer: "near zero on early repetitions of the same sentence".

**Lines 700–704: Orgad "truthfulness information is concentrated at specific token positions, ... probing those positions substantially improves error detection, and ... detectors do not generalise across datasets".** Accurate. Abstract: error detectors "fail to generalize across datasets, implying that ... truthfulness encoding is not universal but rather multifaceted"; the introduction says probes "do not generalize across different tasks" — [arXiv 2410.02707](https://arxiv.org/abs/2410.02707); [HTML](https://arxiv.org/html/2410.02707v4)

**Lines 704–709: McKenzie "cheap first-stage monitor ... last-token readout performs worst while attention-, softmax- and mean-aggregated probes generalise well; best probe recovers only 43% of positives at a 1% FPR; probes suit tiered monitoring where a false alarm costs compute rather than user experience".** Accurate. "our best probe detects only 43% of high-stakes cases at 1% FPR, compared to 55% for the best LLM baseline"; "in tiered monitoring contexts, false positives from the initial filter incur only computational costs"; Attention "works best, though the improvement over Softmax and Mean is small"; probes are about six orders of magnitude cheaper than LLM monitors — [arXiv HTML 2506.10805](https://arxiv.org/html/2506.10805v4); [abstract](https://arxiv.org/abs/2506.10805)

**Lines 710–713: Hedström "use a linear probe to obtain an error-reducing steering direction and calibrate the steering threshold against the probe's error on held-out data, abstaining when the probe is unreliable, so that an intervention fires only under a user-specified tolerance".** Mostly accurate, last clause imprecise. MERA grid-searches α on a calibration set separate from probe training and keeps α only if the gain clears a Bonferroni-corrected Hoeffding bound, P(Δcal(α*) > ε) ≥ 1 − δ; "If no α yields a statistically significant improvement, we abstain from intervention" (δ=0.01, ε=0 in experiments) — [arXiv HTML 2510.13290](https://arxiv.org/html/2510.13290v1). So abstention is triggered by "no significant expected improvement", not by "the probe is unreliable", and the user-specified quantity is a confidence level δ, not an error tolerance.

**Lines 216–222: Li/FineHarm "token-level harmfulness annotations for 29K responses and a streaming monitor that terminates generation once k tokens have been flagged ... tuned by validation F1 ... timeliness as the fraction of the response consumed ... harmful words are spread throughout their responses".** Accurate.
- Generation stops "until there are k harmful tokens being detected", k ∈ {1,…,10} chosen by validation macro-F1; timeliness is the stopping position as a share of the response (about 18% on average); no metric relative to the first harmful token; harmful words are "roughly evenly distributed across the responses" (Fig. 2) — [arXiv HTML 2506.09996](https://arxiv.org/html/2506.09996v3)
- The token labels are produced automatically (moderator majority vote, LLM sentence labels, POS tagging to "label only notional words as harmful words"), not by humans. "Construct token-level annotations" is correct; if the draft later contrasts with human-reviewed onsets, this is worth stating — same source

**Line 223: RAGTruth "annotate hallucinated spans at the word level".** Accurate: "meticulous manual annotations at both the individual case and word levels" — [ACL Anthology](https://aclanthology.org/2024.acl-long.585/)

**Line 225–227: "the transition into harmful generation is itself concentrated at a locatable point in the token sequence [qi]".** Loose. Qi's claim is that safety alignment "adapts a model's generative distribution primarily over only its very first few output tokens" ("shallow safety alignment"), linking prefilling, suffix, decoding and fine-tuning attacks to it — [arXiv 2406.05946](https://arxiv.org/abs/2406.05946); [ICLR blog](https://blog.iclr.cc/2025/04/22/announcing-the-outstanding-paper-awards-at-iclr-2025/). This says the refusal-vs-comply decision is concentrated in the first few tokens; it does not say that harmful content in general begins at a locatable onset. Suggested wording: "and whether a model refuses or complies is decided mostly within the first few output tokens [qi2025shallow]".

**Line 1065: "This differs from [yu], as we use activation similarity directly as the token-level score without training a classifier."** Accurate (Yu trains an MLP on similarity features; see above).

**Lines 173–174, 191–192, 374–376, 749–752: Apertus and Llama model citations.** Correct papers for Apertus-8B-Instruct-2509 and Llama-3.1-8B-Instruct — [arXiv 2509.14233](https://arxiv.org/abs/2509.14233); [arXiv 2407.21783](https://arxiv.org/abs/2407.21783)

### Inferences
- The MERA misattribution on line 297 is the only claim that is wrong in kind; the others are overstatements or ranges that need qualification.
- MERA probes at the "exact" answer token, taken from Orgad, so citing MERA next to Orgad for token selection would be more defensible than citing it for adaptation.
- `\citet` vs `\citep` errors: lines 148, 251, 252 use `\citep` as a sentence subject (should be `\citet`); line 297 uses `\citet` three times parenthetically (should be `\citep`).

### Gaps
- The Duan appendix describing classifier training data and labels (which sentences are negative, train/test split) was not in the HTML I could read; whether Table 2 is held-out accuracy is unknown.
- I could not confirm whether WSC's Table 5 test set is balanced; the paper only states the training set was rebalanced 1:1.
- Yu's evaluation unit (per prompt vs per token) is not explicitly defined in the paper.

## 3. Which cited works use response-, chunk/sentence- or token-level evaluation, and with which metrics?

### Takeaway
Every detection paper cited reports aggregate classification quality at the response, sentence or chunk level (accuracy, F1, AUC/AUROC, TPR at fixed FPR). Only Duan et al. and Li et al. report any timing, and neither measures alarm timing relative to an annotated onset under a fixed false-alarm rate. This supports the draft's framing.

### Cited Findings
- Duan et al.: sentence-level (statement loops) or token-level (numeral loops) classifier with ACC/F1/AUC; response-level early-warning metrics EDR and FPR, plus average lead in sentences (ASLT) and tokens (ATLT); CUSUM threshold h = α·S_max calibrated on 50 non-loop instances; balanced test sets, greedy decoding — [arXiv HTML 2601.05693](https://arxiv.org/html/2601.05693v1)
- Xie et al. (WSC): chunk-level detection (chunk ends at "\n\n"), accuracy and AUROC; downstream task accuracy and token savings for the chopping intervention — [arXiv HTML 2511.00536](https://arxiv.org/html/2511.00536v1)
- Yu et al.: online detector, accuracy 0.9524, F1 0.87, FPR 0.0259, recall 0.8288, averaged over six open models; unit not defined, likely per generation — [arXiv HTML 2503.00416](https://arxiv.org/html/2503.00416)
- McKenzie et al.: sample-level labels; AUROC and TPR at 1% FPR; no token-level or timing evaluation — [arXiv HTML 2506.10805](https://arxiv.org/html/2506.10805v4)
- Orgad et al.: AUC of error detectors over generated answers, probe position and layer swept — [arXiv HTML 2410.02707](https://arxiv.org/html/2410.02707v4)
- Hedström et al.: per-prompt task accuracy, RMSE of the probe, and a Steering Performance Impact score in [−1, 1] — [arXiv HTML 2510.13290](https://arxiv.org/html/2510.13290v1)
- Li et al. (FineHarm/SCM): response-level precision, recall, F1 and macro-F1 (above 0.95); token predictions turned into response decisions via "Delay-k"; timeliness as the share of the response seen before stopping (about 18%) — [arXiv HTML 2506.09996](https://arxiv.org/html/2506.09996v3); [abstract](https://arxiv.org/abs/2506.09996)
- Pipis et al.: no detector; loop prevalence with a lexical rule ("some n-gram appearing at least k times", n=30, k=20) — [arXiv HTML 2512.12895](https://arxiv.org/html/2512.12895v2)
- Welleck et al.: sequence-level seq-rep-4 and token-level rep/ℓ, wrep/ℓ, uniq as corpus statistics — [ar5iv 1908.04319](https://ar5iv.labs.arxiv.org/html/1908.04319)
- Holtzman et al.: corpus-level repetition %, perplexity, Self-BLEU, HUSE across decoding methods — [ar5iv 1904.09751](https://ar5iv.labs.arxiv.org/html/1904.09751)

### Inferences
- Duan is the closest prior evaluation to the draft (lead time in tokens plus a calibrated FPR). The differences worth stating are that Duan's FPR is measured on balanced test sets with greedy decoding, the threshold is a multiple of the maximum normal CUSUM statistic and not a fixed false-alarm budget, and "early detection" is relative to verbatim repetition, not to an annotated onset. The draft's related-work paragraph already says most of this.
- Li's per-response false positives are never fixed in advance; they report the precision/recall trade-off as k varies (precision about 0.84 to 0.98, recall about 0.99 to 0.91) — [arXiv HTML 2506.09996](https://arxiv.org/html/2506.09996v3)

### Gaps
- Whether Duan's EDR/FPR are computed per response is implied but not stated explicitly in the text I read.

## 4. Missing citations and uncited claims

### Takeaway
No cited key is missing from the bib. Uncited claims worth fixing are the rep/ℓ baseline (Welleck), the LRS baseline (Manber and Myers, already in the bib), the LLM-as-judge protocol, the seven prompt-source datasets (only hyperlinked), and LayerNorm and CUSUM as named methods.

### Cited Findings
- rep/ℓ is defined in Welleck et al. ("fraction of next-token (top-1) predictions that occur in the previous ℓ tokens") — [ar5iv 1908.04319](https://ar5iv.labs.arxiv.org/html/1908.04319)
- LLM-as-a-judge: Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023 Datasets and Benchmarks — [arXiv 2306.05685](https://arxiv.org/abs/2306.05685)
- Prompt sources with papers: DeepMath-103K (He et al. 2025) — [arXiv 2504.11456](https://arxiv.org/abs/2504.11456); medical-o1-verifiable-problem comes from HuatuoGPT-o1 (Chen et al. 2024) — [arXiv 2412.18925](https://arxiv.org/abs/2412.18925); Llama-Nemotron post-training data (Bercovich et al. 2025) — [arXiv 2505.00949](https://arxiv.org/abs/2505.00949)
- Pipis also supports the low-temperature prevalence claim on line 255 and could be added there: looping is common "at low temperatures or with greedy decoding" — [arXiv 2512.12895](https://arxiv.org/abs/2512.12895)

### Inferences
- Uncited claims found in the read sections:
  - Line 132 (abstract): "representations are known to carry signals of repetitive degeneration and ... anticipate its onset". Abstracts usually carry no citations; the intro covers it.
  - Lines 316–318 and 961–988: rep/ℓ baseline, cite welleck2020neural.
  - Lines 314–315 and 1017–1043: LRS baseline; manber1993suffix is in the bib but not cited. Cite it if the implementation uses a suffix array.
  - Lines 267, 380, 785: LLM judge; cite Zheng et al. 2023 for the method and the Claude Sonnet 5 model card or system card for the judge model (I did not look for that card).
  - Lines 149, 684: CUSUM is credited to Duan; the original is Page (1954), "Continuous Inspection Schemes", Biometrika 41(1/2):100–115 (from knowledge, not verified in this session).
  - Line 292 and eq. 1078–1083: LayerNorm, Ba, Kiros and Hinton (2016), arXiv:1607.06450 (from knowledge, not verified).
  - Lines 731–746: seven datasets only hyperlinked. Workshop reviewers usually expect dataset citations. Beyond the three verified above, NuminaMath, MathArena (AIME 2025), open-r1 codeforces and allenai IF_sft_data_verified (likely Tülu 3) have papers or dataset cards I did not verify in this session.
  - Lines 645–647: "seq-rep-n ... remains the standard measure of repetition" is an opinion without support; soften to "a widely used measure".
  - Lines 179–184: "AUC is not vulnerable to this imbalance, since it is threshold-free and rank-based" is a standard fact; a textbook citation such as Fawcett (2006) is optional.
  - Line 714–716: "LoRA ... the standard way to modify a model's representations at low cost" is an opinion; fine with the existing citation.
- If Mistral-7B-Instruct results are added, the model needs a citation (Jiang et al. 2023, arXiv:2310.06825, from knowledge).
- The four unused bib entries (ProcessBench, WildGuard, Baker et al., Manber and Myers) look like leftovers from a broader "other targets" paragraph. Either cite them (ProcessBench gives step-level first-error labels, which fits the onset argument next to FineHarm and RAGTruth) or remove them.
- Side observations outside the citation audit: the abstract repeats "For this purpose, we introduce ..." twice and describes the protocol twice; the draft contains "rather than" in several places (lines 209, 221, 659, 682, 706, 906), which the author's style rules forbid.

### Gaps
- I did not check every sentence of main.tex outside the requested line ranges (results sections 380–630 and appendices after 1110), so uncited claims there are not covered.
- Page (1954), Ba et al. (2016), Jiang et al. (2023) and the remaining dataset papers were not verified online in this session.
