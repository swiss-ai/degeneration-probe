# Dataset inspection — `lorenzo0312/degeneration-probe-instruct-token-level-balanced`

HF dataset used by `configs/dataset/repetition_balanced_layer_sweep.yaml`.
Cache revision inspected: `55831e27b7609554a4f3d40a3a6f18e481f24d4c` (newest).

One row = one generated sample = one unique `prompt_id` (in every split, `n_rows == n_unique_prompts`).
There is **no prompt_id overlap** between train, validation, and test (all pairwise intersections = 0).

## Totals

| split      | rows  | unique prompts |
|------------|-------|----------------|
| train      | 2379  | 2379           |
| validation | 299   | 299            |
| test       | 295   | 295            |
| **all**    | 2973  | 2973           |

## Prompts per source dataset

Counts are unique prompts (= rows) per `source_dataset` value.

| source_dataset                                       | train | val | test | total | total % |
|------------------------------------------------------|------:|----:|-----:|------:|--------:|
| zwhe99/DeepMath-103K (math)                          | 654   | 83  | 78   | 815   | 27.4%   |
| AI-MO/NuminaMath-1.5 (math)                          | 610   | 88  | 76   | 774   | 26.0%   |
| allenai/IF_sft_data_verified (instr. following)      | 563   | 63  | 70   | 696   | 23.4%   |
| nvidia/Llama-Nemotron-Post-Training-Dataset (mixed)  | 459   | 52  | 60   | 571   | 19.2%   |
| FreedomIntelligence/medical-o1-verifiable-problem    | 93    | 13  | 11   | 117   |  3.9%   |
| **total**                                            | 2379  | 299 | 295  | 2973  | 100%    |

## Per-split composition (% of split)

| source_dataset                                       | train % | val %  | test % |
|------------------------------------------------------|--------:|-------:|-------:|
| zwhe99/DeepMath-103K                                 | 27.5%   | 27.8%  | 26.4%  |
| AI-MO/NuminaMath-1.5                                 | 25.6%   | 29.4%  | 25.8%  |
| allenai/IF_sft_data_verified                         | 23.7%   | 21.1%  | 23.7%  |
| nvidia/Llama-Nemotron-Post-Training-Dataset          | 19.3%   | 17.4%  | 20.3%  |
| FreedomIntelligence/medical-o1-verifiable-problem    |  3.9%   |  4.3%  |  3.7%  |

The relative proportions are consistent across splits (no source is over- or under-represented in val/test vs train by more than ~4 percentage points).

If we collapse the two math sources, **math is ~53% of every split**:
- train: 53.1%
- val:   57.2%
- test:  52.2%

## Notes

- `source_dataset` is read directly from the HF dataset schema (string column).
- Counts are at the *prompt* level (one row per generated rollout). Token-level statistics (e.g. number of training tokens, % of degenerating tokens) would differ — they live in the `chunk_summary` field of each row and are not aggregated here.
- Script used to produce this report: pyarrow read of the cached `.arrow` files at
  `/capstor/scratch/cscs/$USER/feature-probes/hf_cache/datasets/lorenzo0312___degeneration-probe-instruct-token-level-balanced/default/0.0.0/<rev>/`.

---

# Appendix A — Per-layer per-source probe metrics

Source: layer sweep `20260515_180522` (layers 10, 14, 18, 22, 24, 26, 28, 29, 30, 31).
Per-token predictions from `eval_results/{repetition_validation,repetition_test}/regression_predictions.npy` were grouped by `source_dataset` via the per-row `chunk_summary` lengths (sum matches the dumped array exactly, verified for every split).

Binary metrics (AUC, F1, accuracy) use the degeneration threshold = **0.8** on the regression target, matching the training config.

Source-name shorthand:
- **DeepMath** = `zwhe99/DeepMath-103K`
- **NuminaMath** = `AI-MO/NuminaMath-1.5`
- **IF_sft** = `allenai/IF_sft_data_verified`
- **Llama-Nemotron** = `nvidia/Llama-Nemotron-Post-Training-Dataset`
- **Medical-o1** = `FreedomIntelligence/medical-o1-verifiable-problem`

## A.0 Per-source token counts and base rates

This is constant across layers; it tells us what each per-source slice is actually measuring.

| source | val n_tokens | val pos_rate (≥0.8) | test n_tokens | test pos_rate (≥0.8) |
|---|---:|---:|---:|---:|
| DeepMath        | 164,653 | 0.653 | 163,335 | 0.737 |
| NuminaMath      | 146,313 | 0.632 | 139,201 | 0.650 |
| IF_sft          | 164,147 | 0.793 | 164,122 | 0.817 |
| Llama-Nemotron  |  60,640 | 0.439 |  77,995 | 0.405 |
| Medical-o1      |   4,687 | 0.000 |   3,910 | 0.000 |

**Important caveat for Medical-o1**: 0% of its tokens cross the 0.8 threshold in either split, so binary metrics (AUC, F1, precision, recall) are degenerate — AUC is undefined (shown as `—`) and F1 = 0 by `zero_division=0` convention. Only the regression metrics (RMSE, Pearson, Spearman) are meaningful for this source.

## A.1 Validation split

### AUC (binary, threshold 0.8)

| layer | DeepMath | NuminaMath | IF_sft | Llama-Nemotron | Medical-o1 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.9857 | 0.9970 | 0.9938 | 0.9971 | — |
| 14 | 0.9883 | 0.9972 | 0.9912 | 0.9985 | — |
| 18 | 0.9881 | 0.9976 | 0.9895 | 0.9992 | — |
| 22 | 0.9873 | 0.9975 | 0.9912 | 0.9984 | — |
| 24 | 0.9901 | 0.9977 | 0.9913 | 0.9982 | — |
| 26 | 0.9889 | 0.9970 | 0.9934 | 0.9982 | — |
| 28 | 0.9885 | 0.9982 | 0.9944 | 0.9986 | — |
| 29 | 0.9880 | 0.9982 | 0.9945 | 0.9986 | — |
| 30 | 0.9885 | 0.9977 | 0.9944 | 0.9987 | — |
| 31 | 0.9883 | 0.9969 | 0.9936 | 0.9980 | — |

### F1 @ 0.8

| layer | DeepMath | NuminaMath | IF_sft | Llama-Nemotron | Medical-o1 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.9539 | 0.9823 | 0.9845 | 0.9659 | 0.0000 |
| 14 | 0.9545 | 0.9816 | 0.9698 | 0.9775 | 0.0000 |
| 18 | 0.9583 | 0.9852 | 0.9692 | 0.9886 | 0.0000 |
| 22 | 0.9630 | 0.9847 | 0.9741 | 0.9867 | 0.0000 |
| 24 | 0.9666 | 0.9831 | 0.9735 | 0.9830 | 0.0000 |
| 26 | 0.9673 | 0.9682 | 0.9747 | 0.9853 | 0.0000 |
| 28 | 0.9648 | 0.9820 | 0.9764 | 0.9900 | 0.0000 |
| 29 | 0.9611 | 0.9777 | 0.9746 | 0.9873 | 0.0000 |
| 30 | 0.9591 | 0.9772 | 0.9755 | 0.9869 | 0.0000 |
| 31 | 0.9688 | 0.9715 | 0.9782 | 0.9838 | 0.0000 |

### RMSE

| layer | DeepMath | NuminaMath | IF_sft | Llama-Nemotron | Medical-o1 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.0517 | 0.0555 | 0.0575 | 0.0685 | 0.1149 |
| 14 | 0.0514 | 0.0554 | 0.0558 | 0.0684 | 0.1163 |
| 18 | 0.0490 | 0.0551 | 0.0552 | 0.0663 | 0.1193 |
| 22 | 0.0484 | 0.0540 | 0.0545 | 0.0681 | 0.1146 |
| 24 | 0.0486 | 0.0541 | 0.0537 | 0.0711 | 0.1194 |
| 26 | 0.0471 | 0.0541 | 0.0545 | 0.0670 | 0.1185 |
| 28 | 0.0476 | 0.0539 | 0.0540 | 0.0651 | 0.1182 |
| 29 | 0.0472 | 0.0545 | 0.0548 | 0.0660 | 0.1183 |
| 30 | 0.0473 | 0.0544 | 0.0531 | 0.0650 | 0.1234 |
| 31 | 0.0490 | 0.0573 | 0.0576 | 0.0711 | 0.1280 |

### Pearson

| layer | DeepMath | NuminaMath | IF_sft | Llama-Nemotron | Medical-o1 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.9539 | 0.9633 | 0.9624 | 0.9830 | 0.8180 |
| 14 | 0.9543 | 0.9633 | 0.9638 | 0.9828 | 0.8247 |
| 18 | 0.9585 | 0.9637 | 0.9643 | 0.9836 | 0.8164 |
| 22 | 0.9593 | 0.9652 | 0.9649 | 0.9830 | 0.8284 |
| 24 | 0.9589 | 0.9650 | 0.9660 | 0.9815 | 0.8149 |
| 26 | 0.9616 | 0.9652 | 0.9651 | 0.9834 | 0.8127 |
| 28 | 0.9609 | 0.9655 | 0.9657 | 0.9843 | 0.8155 |
| 29 | 0.9617 | 0.9647 | 0.9651 | 0.9839 | 0.8191 |
| 30 | 0.9613 | 0.9649 | 0.9671 | 0.9845 | 0.8140 |
| 31 | 0.9583 | 0.9607 | 0.9603 | 0.9816 | 0.7850 |

### Spearman

| layer | DeepMath | NuminaMath | IF_sft | Llama-Nemotron | Medical-o1 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.9599 | 0.9690 | 0.9770 | 0.9247 | 0.7684 |
| 14 | 0.9571 | 0.9690 | 0.9807 | 0.9196 | 0.7869 |
| 18 | 0.9617 | 0.9707 | 0.9690 | 0.9219 | 0.7586 |
| 22 | 0.9626 | 0.9724 | 0.9784 | 0.9201 | 0.7625 |
| 24 | 0.9649 | 0.9718 | 0.9730 | 0.9139 | 0.7473 |
| 26 | 0.9665 | 0.9737 | 0.9810 | 0.9201 | 0.7528 |
| 28 | 0.9673 | 0.9735 | 0.9752 | 0.9252 | 0.7563 |
| 29 | 0.9713 | 0.9728 | 0.9814 | 0.9239 | 0.7594 |
| 30 | 0.9685 | 0.9729 | 0.9796 | 0.9267 | 0.7582 |
| 31 | 0.9655 | 0.9681 | 0.9719 | 0.9150 | 0.6831 |

## A.2 Test split

### AUC (binary, threshold 0.8)

| layer | DeepMath | NuminaMath | IF_sft | Llama-Nemotron | Medical-o1 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.9752 | 0.9942 | 0.9965 | 0.9987 | — |
| 14 | 0.9897 | 0.9945 | 0.9979 | 0.9991 | — |
| 18 | 0.9891 | 0.9956 | 0.9978 | 0.9991 | — |
| 22 | 0.9862 | 0.9958 | 0.9974 | 0.9990 | — |
| 24 | 0.9866 | 0.9956 | 0.9971 | 0.9990 | — |
| 26 | 0.9862 | 0.9952 | 0.9970 | 0.9988 | — |
| 28 | 0.9868 | 0.9954 | 0.9973 | 0.9990 | — |
| 29 | 0.9858 | 0.9956 | 0.9976 | 0.9989 | — |
| 30 | 0.9852 | 0.9958 | 0.9975 | 0.9989 | — |
| 31 | 0.9806 | 0.9949 | 0.9966 | 0.9987 | — |

### F1 @ 0.8

| layer | DeepMath | NuminaMath | IF_sft | Llama-Nemotron | Medical-o1 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.9714 | 0.9542 | 0.9854 | 0.9921 | 0.0000 |
| 14 | 0.9678 | 0.9469 | 0.9913 | 0.9919 | 0.0000 |
| 18 | 0.9728 | 0.9577 | 0.9895 | 0.9932 | 0.0000 |
| 22 | 0.9750 | 0.9593 | 0.9910 | 0.9937 | 0.0000 |
| 24 | 0.9768 | 0.9564 | 0.9904 | 0.9937 | 0.0000 |
| 26 | 0.9762 | 0.9600 | 0.9920 | 0.9938 | 0.0000 |
| 28 | 0.9809 | 0.9555 | 0.9926 | 0.9938 | 0.0000 |
| 29 | 0.9799 | 0.9500 | 0.9914 | 0.9937 | 0.0000 |
| 30 | 0.9698 | 0.9558 | 0.9911 | 0.9938 | 0.0000 |
| 31 | 0.9783 | 0.9594 | 0.9926 | 0.9939 | 0.0000 |

### RMSE

| layer | DeepMath | NuminaMath | IF_sft | Llama-Nemotron | Medical-o1 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.0505 | 0.0559 | 0.0522 | 0.0768 | 0.0867 |
| 14 | 0.0498 | 0.0565 | 0.0515 | 0.0778 | 0.0920 |
| 18 | 0.0490 | 0.0535 | 0.0508 | 0.0765 | 0.0871 |
| 22 | 0.0482 | 0.0537 | 0.0509 | 0.0765 | 0.0894 |
| 24 | 0.0480 | 0.0547 | 0.0518 | 0.0770 | 0.0990 |
| 26 | 0.0469 | 0.0535 | 0.0509 | 0.0751 | 0.0938 |
| 28 | 0.0480 | 0.0532 | 0.0504 | 0.0750 | 0.0974 |
| 29 | 0.0486 | 0.0545 | 0.0507 | 0.0756 | 0.0967 |
| 30 | 0.0486 | 0.0538 | 0.0503 | 0.0758 | 0.0939 |
| 31 | 0.0495 | 0.0546 | 0.0522 | 0.0796 | 0.0986 |

### Pearson

| layer | DeepMath | NuminaMath | IF_sft | Llama-Nemotron | Medical-o1 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.9533 | 0.9646 | 0.9778 | 0.9760 | 0.5173 |
| 14 | 0.9549 | 0.9637 | 0.9780 | 0.9755 | 0.4685 |
| 18 | 0.9570 | 0.9673 | 0.9783 | 0.9766 | 0.5314 |
| 22 | 0.9578 | 0.9672 | 0.9783 | 0.9761 | 0.4941 |
| 24 | 0.9579 | 0.9661 | 0.9777 | 0.9757 | 0.3971 |
| 26 | 0.9599 | 0.9674 | 0.9783 | 0.9771 | 0.4773 |
| 28 | 0.9583 | 0.9677 | 0.9786 | 0.9771 | 0.4455 |
| 29 | 0.9572 | 0.9660 | 0.9784 | 0.9768 | 0.4793 |
| 30 | 0.9570 | 0.9669 | 0.9789 | 0.9767 | 0.5053 |
| 31 | 0.9553 | 0.9658 | 0.9774 | 0.9740 | 0.3590 |

### Spearman

| layer | DeepMath | NuminaMath | IF_sft | Llama-Nemotron | Medical-o1 |
|---|---:|---:|---:|---:|---:|
| 10 | 0.9414 | 0.9651 | 0.9591 | 0.9502 | 0.5715 |
| 14 | 0.9458 | 0.9564 | 0.9603 | 0.9516 | 0.5059 |
| 18 | 0.9580 | 0.9652 | 0.9602 | 0.9506 | 0.5528 |
| 22 | 0.9570 | 0.9665 | 0.9610 | 0.9486 | 0.5224 |
| 24 | 0.9602 | 0.9695 | 0.9575 | 0.9483 | 0.4161 |
| 26 | 0.9629 | 0.9678 | 0.9641 | 0.9527 | 0.5128 |
| 28 | 0.9625 | 0.9713 | 0.9602 | 0.9507 | 0.4902 |
| 29 | 0.9644 | 0.9694 | 0.9657 | 0.9494 | 0.5268 |
| 30 | 0.9562 | 0.9690 | 0.9653 | 0.9485 | 0.5372 |
| 31 | 0.9533 | 0.9663 | 0.9631 | 0.9435 | 0.3940 |

## A.3 Observations

- **Per-source ranking is preserved across layers**: Llama-Nemotron is always the easiest (AUC ≈ 0.998–0.999), DeepMath the hardest (AUC ≈ 0.985–0.990), with NuminaMath and IF_sft in between. The relative difficulty of the sources is a property of the data, not the layer.
- **Layer-to-layer variation within a source is small**: AUC swings within a source are typically < 0.005, RMSE within ~0.005. This matches the global picture — different layers produce essentially equivalent probes.
- **Medical-o1 is an outlier**: positive rate = 0 means it carries no degenerating tokens at all. Pearson is much lower (0.36–0.83) because the target distribution is squeezed near 0 and the probe's predictions there are noisier in relative terms; RMSE is also ~2× the other sources. Treat Medical-o1 metrics as a "false-positive stress test" rather than a probe-quality benchmark.
- **Spearman drops on Llama-Nemotron** (~0.92–0.95) compared to other sources (~0.96–0.98), despite its top AUC. The probe ranks positives vs negatives well there but is less monotonic in the mid-range scores — likely because the Nemotron rollouts have more bimodal degeneration patterns (large clean stretches + abrupt repetition).
- **Layer 31 is the only consistent loser**: across nearly every source × split × metric it ranks worst or near-worst, confirming the global RMSE bump seen in the main report. Layers 24–30 are tied; layers 10–18 are essentially equivalent on probe quality, with their advantage being preservation of `lm_loss`.

Raw per-source per-layer per-split metrics are saved as JSON:
`/iopsstor/scratch/cscs/lbaggi/degeneration-probe/per_layer_per_source_metrics.json`.
