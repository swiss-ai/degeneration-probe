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

CONFIG="${CONFIG:-configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml}"

# "claude_agent_sdk" (default) rotates personal Pro/Max subscription OAuth
# tokens below. "anthropic" is pay-per-token against a single API key (the
# AI-center-issued key at ~/keys/.anthropic_api_key_ai_center, see
# AnthropicBackend) -- no per-account usage window, so no rotation needed,
# just one run. Override with e.g. --export=ALL,BACKEND=anthropic
BACKEND="${BACKEND:-claude_agent_sdk}"

echo "Commit: $(git rev-parse HEAD)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Job: $SLURM_JOB_ID  Config: $CONFIG  Backend: $BACKEND"

# Generation runs as wall-clock-limited waves and can stop with a domain part
# done. This stage would not fail on that, it would succeed on partial input --
# so check first and refuse rather than produce a quietly wrong result.
# Explicit exit: this script deliberately runs without `set -e` so an exhausted
# account does not abort the rotation, which also means a failing check here
# would otherwise be ignored.
.venv/bin/python scripts/assert_generation_complete.py --config "$CONFIG" || exit 1

if [[ "$BACKEND" != "claude_agent_sdk" ]]; then
    python -m degeneration_probe.dataset_gen.llm_judge \
        --config "$CONFIG" \
        --backend "$BACKEND"
    echo "Judging run done."
    exit 0
fi

# Rotate across accounts, in order, resuming the same results_claude_agent_sdk.parquet
# manifest each time -- llm_judge.py already skips rows marked "ok" and stops
# cleanly (exit 0) on UsageExhausted, so each iteration below just continues
# wherever the previous account's usage ran out.
# Override the rotation order with e.g. --export=ALL,TOKEN_ORDER="lorenzobaggi moritzreihs marcodenegri lucasartori"
# -- lets two judge jobs for different datasets run concurrently without both
# starting from the same account and colliding on its usage window.
TOKEN_NAMES=(${TOKEN_ORDER:-marcodenegri moritzreihs lucasartori lorenzobaggi})

remaining() {
    python3 - <<PY
import pandas as pd
from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig

config = DatasetGenConfig.from_yaml("$CONFIG")
sample = pd.read_parquet(paths.llm_judge_sample_path(config))
sample_keys = set(zip(sample["prompt_id"], sample["rollout_idx"]))
results_path = paths.llm_judge_results_path(config, "claude_agent_sdk")
done = 0
if results_path.exists():
    results = pd.read_parquet(results_path)
    ok = results[results["status"] == "ok"]
    done = len(set(zip(ok["prompt_id"], ok["rollout_idx"])) & sample_keys)
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
