# ONWARD: early-warning probes for repetitive degeneration

This repository holds the code behind *Degeneration Detection Is Not Early Warning: False-Alarm
Evaluation with ONWARD*. It covers every step of the paper: building a corpus of sampled responses with
annotated degeneration onsets, training linear probes on a generator's hidden states, selecting and scoring
them, evaluating them with the onset-aligned ONWARD metrics, scoring the untrained baselines, reading the
results on human-verified labels, and drawing the paper's tables and figures.

The trained probes and their calibrated thresholds are released on Hugging Face:
[huggingface.co/collections/marcodene/onward-degeneration-probes](https://huggingface.co/collections/marcodene/onward-degeneration-probes).

## Contents

- [Overview](#overview)
- [Setup](#setup)
- [1. Build a dataset](#1-build-a-dataset)
- [2. Train probes](#2-train-probes)
- [3. Select, score and evaluate](#3-select-score-and-evaluate)
- [4. Evaluate a probe on another dataset](#4-evaluate-a-probe-on-another-dataset)
- [5. Baselines](#5-baselines)
- [6. Human-verified labels](#6-human-verified-labels)
- [7. Tables and figures](#7-tables-and-figures)
- [8. Reproducing the paper, in order](#8-reproducing-the-paper-in-order)
- [9. Adding a new model](#9-adding-a-new-model)
- [10. Adding a new prompt source](#10-adding-a-new-prompt-source)
- [11. Released probes](#11-released-probes)
- [Tests](#tests)

## Overview

| Stage | What it produces | Main entry points |
|---|---|---|
| Dataset | Prompts, sampled responses, structural labels, judge verdicts, onset labels, cached activations | `degeneration_probe/dataset_gen/`, `cluster/utils/dataset/` |
| Training | One probe per depth (frozen) or one adapted probe per layer, with a checkpoint every 100 steps | `scripts/train_probe.py`, `cluster/train.sbatch`, `cluster/launch_*.sh` |
| Selection | The checkpoint chosen for each depth on validation | `scripts/replay_checkpoints.py`, `scripts/select_checkpoints.py` |
| Scoring and evaluation | One score per token on every split; thresholds calibrated on validation; the ONWARD metrics | `scripts/score_selected_depths.py`, `scripts/score_rollouts.py`, `scripts/evaluate_scores.py` |
| Baselines | Rule-based and confidence-based scorers evaluated the same way | `scripts/score_baselines.py`, `scripts/select_baseline_sweep.py` |
| Human labels | Every test result re-read on human-verified onsets | `scripts/evaluate_human_labels.py`, `cluster/human_*.sbatch` |
| Paper | LaTeX tables and PDF figures | `scripts/rebuttal_tables.py`, `scripts/baseline_sweep_tables.py`, `notebooks/paper_figures.py` |

The paper uses three generators, each with its own dataset build:

| Generator | Hugging Face id | Model config | Dataset build |
|---|---|---|---|
| Apertus | `swiss-ai/Apertus-8B-Instruct-2509` | `configs/model/apertus.yaml` | `degeneration-dataset-apertus-8b-instruct` |
| Llama | `meta-llama/Llama-3.1-8B-Instruct` | `configs/model/llama3p1.yaml` | `degeneration-dataset-llama3p1-8b-instruct` |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.1` | `configs/model/mistral.yaml` | `degeneration-dataset-mistral-7b-instruct-v0p1` |

## Setup

### Python environment

The package needs Python 3.10 to 3.12. Dependencies are pinned in `uv.lock`.

```bash
uv sync                     # creates .venv with the package installed in editable mode
uv run pytest               # optional: run the test suite
```

CPU jobs (labelling, the judge, baselines, evaluation, aggregation, tables) run directly from `.venv`.

### Container for GPU jobs

GPU jobs (generation, activation caching, training, live scoring) run in a container through the Slurm
`--environment` flag. Build the image once from the local tree:

```bash
srun --account=<account> --partition=normal --time=01:30:00 bash cluster/build_local.sh
```

This builds `cluster/Dockerfile.local` with the pins in `cluster/constraints.txt` and writes a squashfs
image. `cluster/env.toml` points at that image. It also mounts the storage, sets the working directory,
places the Hugging Face, uv and W&B caches, and forwards `HF_TOKEN`. Edit the image path, mounts and
cache paths in `cluster/env.toml` for your cluster.

### Credentials

Jobs read credentials from files in `~/keys/`, never from the repository:

| File | Used by |
|---|---|
| `~/keys/.hf_token` | Every job that downloads a model. Needed for gated models such as Llama. |
| `~/keys/.wandb_key` | Training, if W&B logging is on. Falls back to `~/.netrc`. |
| `~/keys/.anthropic_oauth_token` or `~/keys/.anthropic_oauth_token_<name>` | The LLM judge with the `claude_agent_sdk` backend. |
| `~/keys/.anthropic_api_key_ai_center` | The LLM judge with the `anthropic` backend. |
| `~/keys/.openrouter_key` | The LLM judge with the `openrouter` backend. |

Training logs to W&B (project `degeneration-probes`) unless you pass `training.wandb.enabled=false`.

### Paths to adapt

The code assumes two locations:

- **Repository:** `/iopsstor/scratch/cscs/$USER/degeneration-probe`. Every sbatch file sets `REPO` to it and
  writes logs to `$REPO/cluster/log/`. Every sbatch file also sets `#SBATCH --account=infra01`.
- **Store:** `/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe`. It holds the dataset
  builds (`degeneration-dataset-<name>/`), all run outputs (`outputs/`), the analysis tables
  (`outputs/analysis/`), human reviews (`human_review/`) and the release folders (`release/`).

To run elsewhere, replace both prefixes everywhere. They appear in the build configs
(`configs/dataset/builds/*.yaml`, `output_root` and `work_root`), the training dataset configs
(`configs/dataset/*.yaml`, `build_root`), the `STORE` constants at the top of the analysis scripts and
notebooks, `degeneration_probe/dataset_gen/config.py` and `manifest.py`, and the sbatch and launcher files
in `cluster/`:

```bash
grep -rl "/capstor/store/cscs/swissai/infra01/users/mdenegri" --include=*.py --include=*.yaml --include=*.sh --include=*.sbatch --include=*.toml .
```

Each run also records its absolute `build_root` in `resolved_config.json`, and evaluation uses it to find
the onset labels. Move a run together with its dataset, or pass `--onset-labels` explicitly.

## 1. Build a dataset

A dataset build is defined by one YAML file in `configs/dataset/builds/`. It names the generator, the
prompt sources, the sampling settings and the output locations. All stages are Python modules taking
`--config <build yaml>`. Without `--config` they default to the Apertus build. Each stage has a Slurm
wrapper in `cluster/utils/dataset/` that reads the build from the `CONFIG` variable:

```bash
export CONFIG=configs/dataset/builds/degeneration-dataset-llama3p1-8b-instruct.yaml
```

The paper's builds share these settings: 3,630 prompts from seven sources, 10 responses per prompt,
temperature 0.7, top-p 0.9, at most 4,096 new tokens, seed 42.

### 1.1 Prompts and splits

```bash
python -m degeneration_probe.dataset_gen.build_prompts --config $CONFIG
```

This loads every source from the Hugging Face Hub, samples `n_prompts` prompts per source with the build
seed, and assigns splits per prompt, so all responses to a prompt share its split. Each in-domain source is
split 70/15/15 into `train`, `val` and `test_indomain`. Held-out sources (`codeforces` and `medical_o1`) go
entirely to `test_heldout_domains`. It writes `prompts/prompts.parquet`, `splits/<split>.jsonl` and
`manifest.json` under the build's `output_root`. It needs network access and has no Slurm wrapper.

The source datasets are not pinned to a revision. To stay comparable with the paper's builds, copy
`prompts/` and `splits/` from an existing build instead of rebuilding them.

| Source (domain) | Hub dataset | Prompts | Split |
|---|---|---|---|
| `deepmath_103k` | `zwhe99/DeepMath-103K` | 600 | in-domain |
| `numinamath_1_5` | `AI-MO/NuminaMath-1.5` | 600 | in-domain |
| `if_sft_data_verified` | `allenai/IF_sft_data_verified` | 600 | in-domain |
| `llama_nemotron` | `nvidia/Llama-Nemotron-Post-Training-Dataset` (SFT, code) | 600 | in-domain |
| `aime_2025` | `MathArena/aime_2025` | 30 | in-domain |
| `medical_o1` | `FreedomIntelligence/medical-o1-verifiable-problem` | 600 | held out |
| `codeforces` | `open-r1/codeforces` (verifiable prompts, rating at least 2000) | 600 | held out |

### 1.2 Generation (GPU)

```bash
bash cluster/utils/dataset/submit_generation.sh                     # every domain, as chains of resumable jobs
sbatch --export=ALL,DOMAIN=deepmath_103k,CONFIG=$CONFIG cluster/utils/dataset/generate.sbatch   # one domain
```

The direct command is `python -m degeneration_probe.dataset_gen.generate --config $CONFIG [--domains ...]
[--batch-size 16]`. Every response has its own seed, so generation is reproducible and resumes where it
stopped. The prompt is the model's chat template for a single user turn, with the beginning-of-sequence
token removed and tokenized without special tokens. Each domain is written to
`generations/<domain>/shard_00000.parquet` with the generated token ids, text, per-token entropy, length
and `stop_reason` (`eos` or `length`). `submit_generation.sh` needs an entry for every domain in its
`WAVES` table, which sets how many 12-hour jobs a domain is split into.

Before any later stage, check that generation is complete:

```bash
python scripts/assert_generation_complete.py --config $CONFIG
```

### 1.3 Structural labels (CPU)

```bash
sbatch --export=ALL,CONFIG=$CONFIG cluster/utils/dataset/label.sbatch
# direct: python -m degeneration_probe.dataset_gen.label --config $CONFIG --window-size 256 --ttr-ngram 2 --lrs-min-length 10 --workers 64
```

This writes per-token repetition metrics to `labels/<domain>/` and `prompt_stats/`. Training requires this
stage. Run it on all domains at once, since `prompt_stats` only covers the domains of the last run.

### 1.4 LLM judge (CPU and network)

```bash
sbatch --export=ALL,CONFIG=$CONFIG cluster/utils/dataset/judge.sbatch                      # claude_agent_sdk backend
sbatch --export=ALL,CONFIG=$CONFIG,BACKEND=anthropic cluster/utils/dataset/judge.sbatch    # Anthropic API
# direct: python -m degeneration_probe.dataset_gen.llm_judge --config $CONFIG --backend {claude_agent_sdk,anthropic,openrouter} [--model ...]
```

Responses that end with the end-of-sequence token are treated as healthy and are not judged. Every capped
response (`stop_reason == length`) is sent to the judge, `claude-sonnet-5` by default, with the prompt and
the full response. The judge decides whether the response degenerates and, if it does, returns a short
verbatim quotation of where the repetitive pattern begins. A verdict whose quotation is not a verbatim
substring of at least four words counts as failed. The system prompt is in
`degeneration_probe/dataset_gen/llm_judge.py`.

The sample of capped responses is fixed in `llm_judge/calibration_sample.parquet` on the first run; delete
it if the generations change. Verdicts are written to `llm_judge/results_<backend>.parquet`. Reruns skip
rows already judged successfully, so a run that stops on a usage limit can simply be resubmitted. With the
`claude_agent_sdk` backend, `judge.sbatch` rotates through the accounts listed in `TOKEN_ORDER`, reading
`~/keys/.anthropic_oauth_token_<name>` for each.

### 1.5 Onset labels (CPU)

```bash
sbatch --export=ALL,CONFIG=$CONFIG cluster/utils/dataset/onset_labels.sbatch
# direct:
python -m degeneration_probe.dataset_gen.onset_quotes --config $CONFIG
python -m degeneration_probe.dataset_gen.onset_labels --config $CONFIG
```

This locates each quotation in the tokenized response; its first token is the onset. It writes
`onset_labels/onset_labels.parquet`, the table training and evaluation read, with one row per response:
`prompt_id`, `rollout_idx`, `domain`, `split`, `stop_reason`, `num_tokens`, `is_positive`,
`onset_position` (a token index into the generated tokens) and `onset_resolution` (`ok`,
`not_degenerating`, `judge_failed`, `not_found`, ...). Responses the judge could not rule on are left out
of training and evaluation.

### 1.6 Activation cache (GPU, frozen probes only)

```bash
sbatch --export=ALL,CONFIG=$CONFIG cluster/utils/dataset/cache_activations.sbatch    # resubmit until complete
```

This runs the generator once over every response and stores the hidden state of every layer at every
generated token, in float16, one safetensors file per response under `activations_root` (by default
`<output_root>/activations/`), plus a `manifest.parquet`. Slot 0 is the embedding output and slot `l+1` the
output of decoder block `l`. The cache takes 6 to 8 TB per 8B-model build. Only frozen probes read it;
adapted probes run the model live.

### Build layout

```
degeneration-dataset-<name>/
  manifest.json
  prompts/prompts.parquet
  splits/{train,val,test_indomain,test_heldout_domains}.jsonl
  generations/<domain>/shard_00000.parquet
  labels/<domain>/shard_00000.parquet
  prompt_stats/prompt_stats.parquet
  llm_judge/{calibration_sample,results_<backend>}.parquet
  onset_labels/{onset_quote_positions,onset_labels}.parquet
  token_logprobs/                     # written in section 5, for the negative log-probability baseline
  activations/                        # or the build's activations_root
```

Training reads a build through a dataset config in `configs/dataset/<name>.yaml`, which gives
`short_name`, `build_root` (the build directory), `build_config` (the build YAML), the split names, the
tokenization limits and the sampling of negatives.

## 2. Train probes

`scripts/train_probe.py` is a Hydra application. `configs/main.yaml` composes a model config
(`configs/model/`), the training config (`configs/training/degeneration.yaml`) and a dataset config
(`configs/dataset/`). Everything can be overridden on the command line, and `cluster/train.sbatch` passes
its arguments straight through:

```bash
sbatch --time=08:00:00 cluster/train.sbatch model=llama3p1 dataset=degeneration-dataset-llama3p1-8b-instruct training.runtime.seed=42 ...
```

### The paper's recipe

A probe is a layer norm followed by a linear map on the residual stream after one decoder block, with
blocks numbered from 0. All probes in the paper use the same recipe:

| Setting | Override | Value |
|---|---|---|
| Token selection | `training.selection.strategy` / `window_size` / `anchor` | `frontier_window` / `512` / `centered` |
| Positive share per batch | `training.selection.positive_fraction` | `0.25` |
| Target | `training.label.family` / `horizon` | `frontier_hard` / `256` (positive from 256 tokens before the onset) |
| Loss | `training.loss.name` / `training.loss.bce.use_pos_weight` | `bce` / `true` |
| Budget | `training.budget.tokens_per_step` / `training.runtime.max_steps` | `4096` / `3000` |
| Validation and checkpoints | `training.validation.steps` / `training.checkpoint.steps` | every `100` steps, all checkpoints kept |
| Stopping | `training.stopping.enabled` | `false` (the stopping rule is still recorded) |
| Seeds | `training.runtime.seed` | `42`, `43`, `44` |

The two regimes differ as follows:

- **Frozen** (`training.features.regime=cached`, `training.lora.enabled=false`). The probe reads the
  activation cache, so the language model is never loaded. One job trains an independent head at every
  depth from 1 to 31 at once (`training.probe.layers=[1,...,31]`), with batch size 8 and 800 validation
  responses per check.
- **Adapted** (`training.features.regime=adapted`, `training.lora.enabled=true`, `training.lora.layers=all`).
  The model runs live with rank-16 LoRA adapters (alpha 32, dropout 0.05) on every linear projection of the
  attention and MLP sublayers of blocks 0 to `l`, trained jointly with one head at
  `training.probe.layer=l`. The learning rate is `1e-4` for the probe and `1e-5` for the adapters.

The complete override lines are in the launchers, one `sbatch` line per run:

| Launcher | Runs |
|---|---|
| `cluster/launch_frozen_sweep.sh` | Apertus, frozen, 26 training strategies x 3 seeds, all depths. Lines 48-50 are the paper's recipe; the other strategies feed the strategy comparison in the appendix. |
| `cluster/launch_frozen_transfer.sh` | Llama and Mistral, frozen, the paper's recipe x 3 seeds, all depths. |
| `cluster/launch_lora_depth.sh` | Apertus, adapted, the paper's recipe at every layer 1-31 x 3 seeds. |
| `cluster/launch_lora_depth_transfer.sh` | Llama and Mistral, adapted, the same. |

All four write to `training.checkpoint.root_dir`, set to the store's `outputs/` directory.
`cluster/resume_timed_out.sh` resubmits runs that hit the time limit, and `cluster/resume_unfinished.sh`
resubmits unfinished frozen runs.

### Run directories

Each run writes to `<root_dir>/<run_name>/<timestamp>/`, with `<run_name>/latest` pointing at the newest
attempt. The run name encodes the dataset, depths, recipe, LoRA scope, seed and a hash of every non-default
setting, for example `apertus-8b-instruct_L1-31_frontier512_hard256_bce_lora-none_s42_9d707c94` or
`apertus-8b-instruct_L15_frontier512_hard256_bce_lora-all_s42_81808a2f`. An attempt holds `run_info.json`,
`resolved_config.json`, `history.parquet`, `selection_history.parquet` (the selection metrics of every
checkpoint and depth, recorded during training) and `checkpoint-<step>/` folders. Frozen checkpoints store
one `layer_NN/` folder per depth; adapted checkpoints store the PEFT adapter next to the probe.

## 3. Select, score and evaluate

### 3.1 Checkpoint selection

For each depth, the selected checkpoint is the one with the highest validation warning coverage
(`Warn_256`) at the 1% false-alarm budget among the checkpoints whose post-onset coverage is at least 0.3.
If no checkpoint reaches that floor, the depth takes the checkpoint with the highest post-onset coverage
and is marked not selectable.

Frozen runs first replay every checkpoint on a fixed set of validation responses
(`configs/dataset/validation_rollouts_<dataset>.csv`); adapted runs use the record written during training:

```bash
sbatch cluster/replay.sbatch <run>/latest 100           # frozen: checkpoint_replay.parquet
python scripts/select_checkpoints.py --run-dir <attempt> --source replay    # frozen
python scripts/select_checkpoints.py --run-dir <attempt> --source history   # adapted
```

Both write `checkpoint_selection.json` in the attempt. Always pass `--run-dir`: without it the script
rewrites the selection of every run under the outputs root.

### 3.2 Scoring and evaluation

The scoring jobs select if needed, score val and both test splits at the selected checkpoints, and
evaluate:

```bash
sbatch cluster/score_selected_all.sbatch <run>/latest       # frozen run, every depth, into selected/layer_NN/
sbatch cluster/score_adapted_selected.sbatch <run>/latest   # adapted run, its one depth
sbatch --dependency=afterok:<train job> cluster/score_adapted_selected.sbatch <train job>   # chained to training
```

For an adapted run, the job runs these steps:

```bash
python scripts/score_rollouts.py --run-dir <attempt> --checkpoint checkpoint-<step> \
    --splits val test_indomain test_heldout_domains --output-dir <attempt>/selected/layer_<NN>
python scripts/evaluate_scores.py --run-dir <attempt>/selected/layer_<NN> --splits val
python scripts/evaluate_scores.py --run-dir <attempt>/selected/layer_<NN> --splits test_indomain test_heldout_domains --reuse-thresholds
```

Frozen runs use `scripts/score_selected_depths.py --run-dir <attempt> --subdir selected` in place of
`score_rollouts.py`, which scores every depth in one pass over the cache.

`evaluate_scores.py` calibrates the thresholds on validation only. For each false-alarm budget beta in
{1%, 5%, 10%}, it sets the lowest threshold at which at most a share beta of healthy validation responses
raise an alarm. It writes them to `decision_thresholds.json`. Test splits must be evaluated with
`--reuse-thresholds` so that they use the validation thresholds. `--persistence m` requires `m` consecutive
flagged tokens before an alarm; the paper uses `m = 1`. Results go to `evaluation/<split>/`:
`view_a_detection.csv`, `view_b_coverage.csv`, `view_c_lead_time.csv`, `view_d_persistence.csv`,
`per_domain_detection.csv`, `rollout_table.csv` and `summary.json`.

The paper's metrics map to these columns:

| Paper | Column | File |
|---|---|---|
| Post-onset coverage | `in_pattern_recall` | `view_b_coverage.csv` |
| Warning coverage `Warn_W` | `warning_recall_W` (W = 32, 64, 128, 256) | `view_b_coverage.csv` |
| Coverage before the warning window | `preonset_coverage_W` | `view_b_coverage.csv` |
| Detection rate | `recall` | `view_a_detection.csv` |
| Realised false-alarm rate | `negative_fpr` | `view_a_detection.csv` |
| Early-warning rate | `fired_before_frontier` | `view_c_lead_time.csv` |
| Lead | `median_offset` | `view_c_lead_time.csv` |
| AUROC | `rollout_auc` | `summary.json` |

### 3.3 Aggregation

```bash
python scripts/aggregate_selected.py            # outputs/analysis/selected_metrics.csv and selected_metrics_by_recipe.csv
sbatch cluster/response_metrics_pool.sbatch val test_indomain test_heldout_domains   # best accuracy: response_metrics_pool.csv
```

`selected_metrics.csv` has one row per run, depth, split and budget. The paper reports each model's frozen
probe at layer 15 (Apertus, Llama) or 5 (Mistral), and its adapted probe at the same layer and at layer 30,
6 or 11. These layers are set in `LAYERS` in `scripts/q3_comparison.py`, `PROBES` in
`scripts/rebuttal_tables.py` and `GENERATORS` in `scripts/export_release.py`.

## 4. Evaluate a probe on another dataset

A trained run always scores the dataset it was trained on. To read a trained probe on another build, score
it with one of the transfer scripts, then evaluate with that build's onset labels:

```bash
# frozen head (reads the target build's activation cache)
python scripts/evaluate_cross_model_transfer.py --run-dir <attempt> \
    --dataset-config configs/dataset/<target>.yaml --checkpoint checkpoint-selected
# adapted run: its LoRA adapters and head, on a build of a model with the same architecture
python scripts/evaluate_lora_transplant.py --run-dir <attempt> --checkpoint checkpoint-<step> \
    --dataset-config configs/dataset/<target>.yaml

ONSETS=<target build_root>/onset_labels/onset_labels.parquet
python scripts/evaluate_scores.py --run-dir <scores dir> --onset-labels $ONSETS --splits val
python scripts/evaluate_scores.py --run-dir <scores dir> --onset-labels $ONSETS --splits test_indomain test_heldout_domains --reuse-thresholds
```

Scores are written to `<attempt>/cross_model/<target>/` or `<attempt>/lora_transplant/<target>/`. The
evaluation calibrates new thresholds on the target build's validation split. To evaluate any scorer on a
new build, it only needs to write one `scores/<split>.parquet` per split in the layout of
`degeneration_probe/evaluation/scores.py`.

## 5. Baselines

All baselines are causal, untrained, and scored per token in [0, 1] by `scripts/score_baselines.py`, then
evaluated with the same protocol as the probes:

| Baseline | Name | Default setting |
|---|---|---|
| Windowed seq-rep-2 | `repetition_trailing` | n = 2, window 256 |
| rep/l | `rep_l` | l = 128, window 128 |
| Repeated-n-gram rule | `ngram_<n>` | any n |
| Windowed entropy | `entropy` | window 128 |
| Windowed negative log-probability | `neglogp` | window 128 |

A setting can be appended to the name: `entropy_w<window>`, `neglogp_w<window>`,
`repetition_trailing_n<n>_w<window>`, `rep_l_l<l>_w<window>`, `ngram_<n>`. The token-cap rule and random
alarms are computed in closed form in `scripts/q3_comparison.py`.

The negative log-probability needs the log-probability of every emitted token, recomputed by the
generator:

```bash
sbatch cluster/token_logprobs.sbatch apertus-8b-instruct swiss-ai/Apertus-8B-Instruct-2509
sbatch cluster/token_logprobs.sbatch llama3p1-8b-instruct meta-llama/Llama-3.1-8B-Instruct
sbatch cluster/token_logprobs.sbatch mistral-7b-instruct-v0p1 mistralai/Mistral-7B-Instruct-v0.1
```

### Baselines at their default settings

```bash
sbatch cluster/baselines.sbatch <dataset>              # every baseline except neglogp
sbatch cluster/baselines.sbatch <dataset> neglogp      # after token_logprobs
```

Results go to `outputs/baselines/<dataset>/<name>/`, scored and evaluated like a probe. Baselines have no
run config, so evaluating one by hand needs `--onset-labels <build>/onset_labels/onset_labels.parquet`.

### Baseline sweep

The paper's comparison reads every baseline at the setting chosen on validation with the probes' rule. The
sweep scores 95 settings per model into `outputs/baselines_sweep/<dataset>/<setting>/`:

| Baseline | Grid |
|---|---|
| `entropy_w`, `neglogp_w` | 8, 16, 32, 64, 128, 256, 512, 1024, 2048 |
| `ngram_` | 4, 6, 8, 12, 16, 24, 30, 48, 64, 96, 128 |
| `repetition_trailing_n{2,3,4}_w` | 32, 64, 128, 256, 512, 1024, 2048, 4096 |
| `rep_l_l{32,64,128,256}_w` | 32, 64, 128, 256, 512, 1024, 2048, 4096 |
| `rep_l_l{512,1024}_w` | 256, 512, 1024, 2048, 4096 |

Each setting is scored and evaluated as follows, run inside a CPU allocation:

```bash
BUILD=$STORE/degeneration-dataset-<dataset>; OUT=$STORE/outputs/baselines_sweep/<dataset>
ONSETS=$BUILD/onset_labels/onset_labels.parquet
python scripts/score_baselines.py --build-root $BUILD --out-dir $OUT --baselines <setting>
python scripts/evaluate_scores.py --run-dir $OUT/<setting> --splits val --onset-labels $ONSETS
python scripts/evaluate_scores.py --run-dir $OUT/<setting> --splits test_indomain test_heldout_domains --reuse-thresholds --onset-labels $ONSETS
```

Then choose each baseline's setting on validation:

```bash
python scripts/select_baseline_sweep.py      # outputs/analysis/baseline_sweep_{all,selected}.csv
```

## 6. Human-verified labels

Every capped response of the two test splits was reviewed by hand. For each, the reviewer confirmed the
judge's label and onset, moved the onset, or changed the label. The review tool and its export and merge scripts are in
`scripts/manual-inspection/`. The returned reviews and adjudications are turned into one label table per
model, `human_review/labels/labels_<dataset>.csv`, with one row per capped test response and its
human-verified `is_positive` and `onset_position`.

The test results are re-read on these labels into an `evaluation_human/` folder next to each
`evaluation/` folder. Validation, and therefore every threshold, stays on the judge's labels. The chain is:

```bash
sbatch cluster/human_eval.sbatch        # label tables, plus the reported probes and the selected sweep baselines
sbatch cluster/human_eval_all.sbatch    # every scored depth and every baseline
sbatch cluster/human_aggregate.sbatch   # selected_metrics_human.csv and response_metrics_pool_human.csv
sbatch cluster/human_analyses.sbatch    # positional control and persistence sweep on human labels
sbatch cluster/human_position.sbatch    # flag share by position relative to the onset (Q4 figure)
```

The underlying script is `scripts/evaluate_human_labels.py` (`--dataset-config`, `--labels-in`,
`--run-dirs`, `--write-evaluation evaluation_human`, `--out`). The analysis and table scripts read the human
versions when given `--human`.

## 7. Tables and figures

The paper source is a separate repository. Its tables are LaTeX files under `paper/tables/rebuttal/`,
written by two scripts.

### Inputs

```bash
python scripts/aggregate_selected.py [--human]
python scripts/q3_comparison.py --sweep [--human]          # outputs/analysis/q3_comparison_sweep[_human].csv
python scripts/position_analysis.py --out-dir $STORE/outputs/analysis/position_analysis [--human]
```

`positional_control.py` and `persistence_sweep.py` take the scorers and probes to compare as arguments.
The paper's lists are in `cluster/human_analyses.sbatch`. Without `--human`, and with
`--out .../positional_control_<split>.csv` or `--out .../persistence_sweep.csv`, the same lines give the
judge-labelled versions. `scripts/random_alarm_reference.py` prints the chance levels of random alarms.

### Tables

```bash
python scripts/baseline_sweep_tables.py --human --out <dir>
python scripts/rebuttal_tables.py --human --out <dir>
```

| Script | Tables |
|---|---|
| `baseline_sweep_tables.py` | `baselines` (probes against the baselines), `probes_judge`, `budget_1pct`, `budget_10pct`, `heldout`, `heldout_human`, `sweep_confidence`, `sweep_ngram`, `sweep_seqrep`, `sweep_repl` |
| `rebuttal_tables.py` | `corpus_splits`, `strategies`, `positional_control`, `rank_transfer`, `persistence`, `per_source` |

Copy the written `.tex` files into `paper/tables/rebuttal/`.

### Figures

`notebooks/paper_figures.py` is a marimo notebook that draws every figure and saves each as PDF and PNG
under `notebooks/figures/paper_figures/`:

```bash
uv run marimo edit notebooks/paper_figures.py                      # interactive
PAPER_FIGURES_LABELS=human uv run python notebooks/paper_figures.py  # headless
```

`PAPER_FIGURES_LABELS` (`human` by default, or `judge`) chooses the labels of the test splits.
`PAPER_FIGURES_SOURCE=auto` (the default) reads live results and falls back to the snapshot in
`notebooks/paper_figures_data.npz` where they are missing.

| Paper figure | Notebook output | Inputs |
|---|---|---|
| Metrics on the three tasks (Q1) | `fig_metrics` | `selected_metrics[_human].csv`, `response_metrics_pool[_human].csv` |
| Probes across models (Q2) | `fig_q2_warning_per_model` | `selected_metrics[_human].csv` |
| Flags by distance from the onset (Q4) | `fig_q4_onset_profile` | `position_analysis[_human]/` |
| False alarms by position (appendix) | `fig_false_alarm_profile` | `position_analysis[_human]/` |
| Judge onset error | `fig_judge_onset_error` | `human_review/labels/` |
| Hidden states of one response | `fig_pca_rollouts` | `outputs/paper_figures/pca_rollouts/` |

The hidden states for the PCA figure are extracted on GPU:

```bash
SPECS=apertus:15 EXTRA="--rollouts numinamath_1_5_00388:2 --extra-adapted-layer 30 --out-dir $STORE/outputs/paper_figures/pca_rollouts" sbatch cluster/extract_pca_states.sbatch
SPECS=llama3p1:15 EXTRA="--rollouts numinamath_1_5_00044:4 --extra-adapted-layer 6 --out-dir $STORE/outputs/paper_figures/pca_rollouts" sbatch cluster/extract_pca_states.sbatch
SPECS=mistral:5 EXTRA="--rollouts numinamath_1_5_00517:9 --extra-adapted-layer 11 --out-dir $STORE/outputs/paper_figures/pca_rollouts" sbatch cluster/extract_pca_states.sbatch
```

## 8. Reproducing the paper, in order

1. **Datasets.** For each of the three build configs in `configs/dataset/builds/`, run section 1: prompts
   (or copy them from an existing build), generation, labels, judge, onset labels and the activation cache.
2. **Frozen probes.** `bash cluster/launch_frozen_sweep.sh` and `bash cluster/launch_frozen_transfer.sh`.
   When a run finishes, `sbatch cluster/replay.sbatch <run>/latest 100`, then
   `sbatch cluster/score_selected_all.sbatch <run>/latest`.
3. **Adapted probes.** `bash cluster/launch_lora_depth.sh` and `bash cluster/launch_lora_depth_transfer.sh`.
   When a run finishes, `sbatch cluster/score_adapted_selected.sbatch <run>/latest`.
4. **Aggregation.** `python scripts/aggregate_selected.py` and
   `sbatch cluster/response_metrics_pool.sbatch val test_indomain test_heldout_domains`.
5. **Baselines.** The token log-probabilities, then the baselines and the baseline sweep for each dataset,
   then `python scripts/select_baseline_sweep.py` (section 5).
6. **Analyses on judge labels.** `q3_comparison.py --sweep`, `position_analysis.py`, `positional_control.py`
   and `persistence_sweep.py` (section 7).
7. **Human labels.** The chain in section 6.
8. **Tables.** `baseline_sweep_tables.py --human` and `rebuttal_tables.py --human`, copied into the paper.
9. **Figures.** The PCA extraction, then `notebooks/paper_figures.py`.

## 9. Adding a new model

1. **Build config.** Copy a file in `configs/dataset/builds/` to
   `degeneration-dataset-<name>.yaml`. Set `model_name` (a Hub id or a local checkpoint), optionally
   `tokenizer_name`, and new `output_root`, `work_root` and `activations_root`. Keep the sources, sampling
   settings and seed to stay comparable with the paper. The model must load with
   `AutoModelForCausalLM.from_pretrained` on one GPU and its tokenizer must have a chat template.
2. **Dataset.** Run section 1 with this config.
3. **Model config.** Add `configs/model/<short>.yaml` with `short_name`, `name`, `tokenizer_name` and
   `dtype`.
4. **Dataset config.** Add `configs/dataset/degeneration-dataset-<name>.yaml` with `short_name`,
   `build_root`, `build_config`, the split names, `tokenization` and `sampling`, following an existing one.
5. **Validation sample for replay.** `python scripts/write_validation_rollouts.py model=<short>
   dataset=degeneration-dataset-<name>`.
6. **Training.** Take the launcher lines of `cluster/launch_frozen_transfer.sh` and
   `cluster/launch_lora_depth_transfer.sh` with `model=<short> dataset=degeneration-dataset-<name>`. For a
   model with N decoder blocks, the frozen sweep covers `training.probe.layers=[1,...,N-1]` and the adapted
   probes take `training.probe.layer` from 1 to N-1. Adapted probes need decoder blocks under
   `model.model.layers` with the usual `q/k/v/o_proj` and `gate/up/down_proj` module names.
7. **Selection, scoring and evaluation** as in section 3, and the baselines as in section 5.

## 10. Adding a new prompt source

Prompt sources exist only as entries of a build config. Add one to `in_domain_sources` (split 70/15/15 into
train, val and test) or to `held_out_sources` (test only):

```yaml
in_domain_sources:
  - name: my_source            # becomes the domain name and the prompt_id prefix
    hf_repo: org/dataset
    hf_subset: null
    hf_split: train
    prompt_field: question     # a string, or a list of {"role", "content"} turns
    n_prompts: 600
    filter_field: rating       # optional: keep rows with row[filter_field] >= min_value
    min_value: 2000
```

The source must load with `datasets.load_dataset(hf_repo, hf_subset, split=hf_split)`. Then:

1. Add the domain to `WAVES` in `cluster/utils/dataset/submit_generation.sh`.
2. Rerun `build_prompts`. Each domain is split on its own, so existing domains keep their prompts and
   splits as long as their upstream data has not changed.
3. Generate the new domain only, then rerun labelling on all domains.
4. Delete `llm_judge/calibration_sample.parquet` so that the judge includes the new capped responses, then
   rerun the judge, the onset labels and the activation cache.

The per-source table of the paper lists its sources in `scripts/rebuttal_tables.py`; add the new one there.

## 11. Released probes

The released probes are the frozen probe, the adapted probe at the same layer and the adapted probe at its
own selected layer for each generator, each at the seed with the best validation `Warn_256` at 1%. Each
Hugging Face repository holds, per probe, `probe.safetensors`, `config.json` (base model, pinned revision
and layer), `thresholds.json` (thresholds at 1%, 5% and 10%), `seeds.json` (all three seeds' metrics) and,
for adapted probes, the PEFT adapter. The model cards contain a self-contained loading example.

`release/onward_probe.py` is a standalone loader that depends only on torch, transformers, peft,
safetensors and huggingface_hub:

```python
from onward_probe import DegenerationMonitor

monitor = DegenerationMonitor.from_pretrained("marcodene/onward-probes-llama-3.1-8b-instruct", subfolder="frozen-L15")
prompt_ids = monitor.build_prompt_ids("Prove that ...")
scores = monitor.score(prompt_ids, generated_ids)   # one probability per generated token
first = monitor.first_alarm(scores, budget=0.05)     # index of the first alarm, or None
```

The release folders are written, checked and documented by:

```bash
python scripts/export_release.py --out $STORE/release       # probes, thresholds and per-seed metrics
python scripts/check_release.py                               # GPU: reloads every probe and compares with the stored scores
python scripts/write_model_cards.py [--labels human]          # model cards
```

## Tests

```bash
uv run pytest
```

Run the suite on a compute node or inside the container. Tests that run torch models can crash on login
nodes without a GPU.
