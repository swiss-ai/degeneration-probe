#!/bin/bash
# Submits cluster/utils/dataset/generate.sbatch as 6 INDEPENDENT per-domain
# wave chains, so no single domain is blocked by the `normal` partition's 12h
# MaxTime, and no domain has to wait on any other domain's progress.
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

set -euo pipefail

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
cd "$REPO"

# domain -> number of waves (worst-case GPU-hours / 12h, rounded up, +1 margin)
declare -A WAVES=(
    [deepmath_103k]=4
    [numinamath_1_5]=3
    [if_sft_data_verified]=3
    [llama_nemotron]=5
    [medical_o1]=2
    [aime_2025]=2
)

for domain in "${!WAVES[@]}"; do
    n_waves=${WAVES[$domain]}
    prev_jobid=""
    echo "=== $domain ($n_waves waves) ==="
    for wave in $(seq 1 "$n_waves"); do
        job_name="degeneration-probe-generate-fullscale-${domain}"
        if [[ -z "$prev_jobid" ]]; then
            jobid=$(sbatch --parsable --job-name="$job_name" --export=ALL,DOMAIN="$domain" cluster/utils/dataset/generate.sbatch)
        else
            jobid=$(sbatch --parsable --job-name="$job_name" --export=ALL,DOMAIN="$domain" --dependency=afterany:"$prev_jobid" cluster/utils/dataset/generate.sbatch)
        fi
        echo "  wave $wave: job $jobid$( [[ -n "$prev_jobid" ]] && echo " (depends on afterany:$prev_jobid)" )"
        prev_jobid=$jobid
    done
done

echo
echo "All chains submitted. Track with: squeue -u $USER --format='%.14i %.9P %.45j %.8T %.10M %R'"
