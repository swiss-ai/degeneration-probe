#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=degeneration-probe-judge
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -uo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
mkdir -p $REPO/cluster/log
cd $REPO

# claude_agent_sdk backend is a CLI subprocess making network calls -- no GPU,
# runs directly against the repo's .venv like label.sbatch.
source .venv/bin/activate

# One of three equally-important dataset configs (see configs/dataset/) --
# override with e.g. --export=ALL,CONFIG=configs/dataset/degeneration-dataset-apertus1p5-capfilter-linear-it8816.yaml
CONFIG="${CONFIG:-configs/dataset/degeneration-dataset-apertus-8b-instruct.yaml}"

echo "Commit: $(git rev-parse HEAD)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Job: $SLURM_JOB_ID  Config: $CONFIG"

# Rotate across accounts, in order, resuming the same results_claude_agent_sdk.parquet
# manifest each time -- llm_judge.py already skips rows marked "ok" and stops
# cleanly (exit 0) on UsageExhausted, so each iteration below just continues
# wherever the previous account's usage ran out.
TOKEN_NAMES=(marcodenegri lucasartori lorenzobaggi moritzreihs)

remaining() {
    python3 - <<PY
import pandas as pd
from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig

config = DatasetGenConfig.from_yaml("$CONFIG")
sample = pd.read_parquet(paths.llm_judge_sample_path(config))
results_path = paths.llm_judge_results_path(config, "claude_agent_sdk")
done = 0
if results_path.exists():
    results = pd.read_parquet(results_path)
    done = int((results["status"] == "ok").sum())
print(len(sample) - done)
PY
}

for name in "${TOKEN_NAMES[@]}"; do
    key_file=~/keys/.anthropic_oauth_token_$name
    if [[ ! -f "$key_file" ]]; then
        echo "=== skipping $name: no key file at $key_file ==="
        continue
    fi

    echo "=== judging with token: $name ($(date)) ==="
    python -m degeneration_probe.dataset_gen.llm_judge \
        --config "$CONFIG" \
        --backend claude_agent_sdk \
        --oauth-key-file "$key_file"

    left=$(remaining)
    echo "=== $left rollout(s) remaining after $name ==="
    if [[ "$left" -le 0 ]]; then
        echo "Calibration sample fully judged. Stopping."
        break
    fi
done

echo "Judging run done."
