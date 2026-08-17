#!/bin/bash
# Submits cluster/utils/dataset/generate.sbatch as one INDEPENDENT wave chain
# per configured domain, so no single domain is blocked by the `normal`
# partition's 12h MaxTime, and no domain has to wait on any other domain's
# progress. Every domain in the build config needs an entry in WAVES below:
# a domain missing from the map is silently never generated.
#
# Each domain gets its own chain of N waves (N sized to that domain's
# worst-case GPU-hour estimate at 4096 tokens / 600 prompts, +1 safety
# margin), submitted via --dependency=afterany:<previous_wave_jobid_for_that_
# domain> -- afterany (not aftercorr) because aftercorr requires the prior
# job to have completed *successfully* (exit code 0), which a job killed by
# hitting the 12h wall-clock limit will NOT satisfy; afterany waits for
# termination regardless of exit status, which is what resumability-based
# resumption across timed-out waves actually needs.
#
# generate.py's existing resumability means each wave just picks up wherever
# the previous one left off (or timed out) for that domain -- no pipeline
# code changes needed. A domain that finishes early becomes a fast no-op in
# its later waves (0 remaining tasks) rather than being skipped outright,
# since waves are pre-submitted as a fixed-length chain up front.
#
# Usage: bash cluster/utils/dataset/submit_generation.sh
#        CONFIG=configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml \
#            bash cluster/utils/dataset/submit_generation.sh

set -euo pipefail

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
cd "$REPO"

CONFIG="${CONFIG:-configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml}"
# Short tag for job names/logs, e.g. "apertus-8b-instruct" from
# "degeneration-dataset-apertus-8b-instruct.yaml" -- keeps job names for
# different datasets' generation runs distinguishable in `squeue`.
DATASET_TAG=$(basename "$CONFIG" .yaml | sed 's/^degeneration-dataset-//')

# domain -> number of waves (worst-case GPU-hours / 12h, rounded up, +1 margin)
declare -A WAVES=(
    [deepmath_103k]=4
    [numinamath_1_5]=3
    [if_sft_data_verified]=3
    [llama_nemotron]=5
    [medical_o1]=2
    [aime_2025]=2
    [codeforces]=4
)

# Fail loudly rather than quietly skipping a domain the build config asks for.
mapfile -t CONFIGURED_DOMAINS < <(
    python - "$CONFIG" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
for source in config["in_domain_sources"] + config["held_out_sources"]:
    print(source["name"])
PY
)
for domain in "${CONFIGURED_DOMAINS[@]}"; do
    if [[ -z "${WAVES[$domain]:-}" ]]; then
        echo "ERROR: $CONFIG configures domain '$domain', which has no wave count in WAVES." >&2
        exit 1
    fi
done

for domain in "${!WAVES[@]}"; do
    n_waves=${WAVES[$domain]}
    prev_jobid=""
    echo "=== $domain ($n_waves waves) ==="
    for wave in $(seq 1 "$n_waves"); do
        job_name="degeneration-probe-generate-${DATASET_TAG}-${domain}"
        if [[ -z "$prev_jobid" ]]; then
            jobid=$(sbatch --parsable --job-name="$job_name" --export=ALL,DOMAIN="$domain",CONFIG="$CONFIG" cluster/utils/dataset/generate.sbatch)
        else
            jobid=$(sbatch --parsable --job-name="$job_name" --export=ALL,DOMAIN="$domain",CONFIG="$CONFIG" --dependency=afterany:"$prev_jobid" cluster/utils/dataset/generate.sbatch)
        fi
        echo "  wave $wave: job $jobid$( [[ -n "$prev_jobid" ]] && echo " (depends on afterany:$prev_jobid)" )"
        prev_jobid=$jobid
    done
done

echo
echo "All chains submitted. Track with: squeue -u $USER --format='%.14i %.9P %.45j %.8T %.10M %R'"
