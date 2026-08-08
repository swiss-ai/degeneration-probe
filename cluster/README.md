# Clariden Cluster

This directory contains the environment definition and job scripts for running
experiments on the [Clariden](https://docs.cscs.ch/clusters/clariden/) cluster at CSCS.
Clariden is part of the [Alps](https://docs.cscs.ch/alps/) platform and uses
GH200 nodes with the
[Container Engine](https://docs.cscs.ch/software/container-engine/) (CE) as the primary
runtime.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Container image built on top of NGC PyTorch 25.06 |
| `build.sh` | Builds the image interactively (run via `srun --pty bash`) |
| `env.toml` | [Environment Definition File](https://docs.cscs.ch/software/container-engine/run/); defines mounts, env vars, NCCL hooks |
| `train.sbatch` | Runs `scripts/train_probe.py` for the degeneration training task |
| `utils/dataset/` | Dataset generation/labeling/judging pipeline job scripts (see below) |

## Environment Design

The environment is split into two layers:

```
Image (.sqsh)  — rebuilt only when pyproject.toml changes
──────────────────────────────────────────────────────────
NGC PyTorch 25.06-py3 (torch, transformers, numpy, ...)
+ extra deps installed via uv (wandb, hydra, peft, ...)

Mounted at runtime via env.toml
──────────────────────────────────────────────────────────
$SCRATCH/degeneration-probe/     ← source code (git pull to update)
~/keys/                  ← HF / W&B tokens
/capstor/scratch/$USER/  ← HF cache, probe checkpoints
```

Code changes never require a rebuild — only `pyproject.toml` changes do, such as adding new dependencies.

> Please note, that this environment contains only the training scripts, not the `annotation` and `generation` pipeline also used in the [old repo](https://github.com/sevdari/hallucination_probes).

## First-Time Setup

**1. Create the image directory with Lustre striping** ([required by CSCS](https://docs.cscs.ch/software/container-engine/run/)):
```bash
mkdir -p $SCRATCH/ce-images
lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M $SCRATCH/ce-images
```

**2. Store your API keys:**
```bash
mkdir -p ~/keys
echo "hf_..." > ~/keys/.hf_token
echo "..." > ~/keys/.wandb_key
echo "hf_..." > ~/keys/.hf_token_write  # optional, only for HF uploads
```

**3. Build the image:**
```bash
# first exec into an interactive job
srun --account=infra01 --partition=normal --time=01:00:00 --pty bash
# run the build
bash cluster/build.sh
```
Unfortunately, this needs to be done in an interactive job, because sbatch does not enable NAT connections from the node.

> Note: optionally, you can 'borrow' an existing `enroot` image, e.g. from here: `/iopsstor/scratch/cscs/tkwiecinski/ce-images/feature-probes+25.06.sqsh`

**Building from a specific branch:**

`build.sh` clones the branch from `origin` (not your local working tree), so push your branch first:
```bash
git push -u origin <branch-name>
```
Then pass the branch name as an argument:
```bash
bash cluster/build.sh <branch-name>
```
This produces `$SCRATCH/ce-images/degeneration-probe+25.06-<branch-name>.sqsh`. Update the `image =` line in `env.toml` to point at that file so `train.sbatch` actually picks it up.

**4. Submit a training job:**
```bash
sbatch cluster/train.sbatch
```
The job runs `scripts/train_probe.py`, whose Hydra defaults already select the
Apertus model, the `degeneration` training profile and the local Apertus dataset
build. Override a training field after the script command when needed, for
example `training.loss.name=mse`.

Dataset-generation jobs live under `cluster/utils/dataset/` and use
`configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml` by default.

## Dataset Generation Pipeline

Jobs under `cluster/utils/dataset/` run in this order, all reading the same
`CONFIG` (defaults to
`configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml`,
override with e.g. `--export=ALL,CONFIG=...`):

| File | Purpose |
|---|---|
| `generate.sbatch` | Generates rollouts for one domain (`DOMAIN` env var required); resumable, meant to be chained across time-limited waves |
| `submit_generation.sh` | Submits `generate.sbatch` as independent per-domain wave chains sized to each domain's GPU-hour estimate, so no domain is blocked by the `normal` partition's 12h wall-clock limit |
| `label.sbatch` | CPU-only labeling (window/TTR/LRS metrics) over generated rollouts; runs against the repo `.venv` directly, no container |
| `judge.sbatch` | LLM-judges rollouts; runs against the repo `.venv` directly. Defaults to the `claude_agent_sdk` backend (rotates personal OAuth tokens); `BACKEND=anthropic` uses a single pay-per-token API key instead |
| `cache_activations.sbatch` | Caches model activations for all domains into a single shared `activations/manifest.parquet`; always runs as one sequential job (not split by domain) to avoid concurrent writers racing on that file |

## Rebuilding vs. Updating Code

| What changed | Action needed |
|---|---|
| Source code | `git pull` inside the running container — no rebuild |
| `pyproject.toml` (new dep) | `bash cluster/build.sh` |
| Base image tag | Update `FROM` in `Dockerfile`, then rebuild |

### Notes

To see some insights about working with the cluster, feel free to browse some [tips](../docs/tips_and_tricks.md).
