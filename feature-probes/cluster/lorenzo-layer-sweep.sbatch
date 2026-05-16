#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --job-name=lorenzo-layer-sweep
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/feature-probes/cluster/log/%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/degeneration-probe/feature-probes/cluster/log/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --array=0-2

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe/feature-probes
mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
echo "Array job: $SLURM_ARRAY_JOB_ID  task: $SLURM_ARRAY_TASK_ID"

export HF_TOKEN="$(cat ~/keys/.hf_token)"
export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"
[[ -f ~/keys/.hf_token_write ]] && export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"

# SWEEP_TIMESTAMP is set by submit_layer_sweep.sh and exported to all array tasks
# so the 10 probes share one timestamp dir. Fallback: array job id.
SWEEP_TIMESTAMP="${SWEEP_TIMESTAMP:-jid_${SLURM_ARRAY_JOB_ID}}"
export PROBE_OUTPUT_DIR="/capstor/scratch/cscs/$USER/feature-probes/output/$SWEEP_TIMESTAMP"
mkdir -p "$PROBE_OUTPUT_DIR"
echo "SWEEP_TIMESTAMP=$SWEEP_TIMESTAMP"
echo "PROBE_OUTPUT_DIR=$PROBE_OUTPUT_DIR"

# Layer sweep definition: 10 probes, distributed 4+4+2 across array tasks 0,1,2
LAYERS=(10 14 18 22 24 26 28 29 30 31)
TASKS_PER_JOB=4
START=$(( SLURM_ARRAY_TASK_ID * TASKS_PER_JOB ))
TOTAL=${#LAYERS[@]}

# Build a wrapper that runs up to 4 probes in parallel on 4 GPUs inside the container
TMPSCRIPT=$REPO/cluster/log/sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sh
cat > $TMPSCRIPT << 'SCRIPT_EOF'
#!/bin/bash
set -euo pipefail
SCRIPT_EOF

for local_gpu in 0 1 2 3; do
    idx=$(( START + local_gpu ))
    [[ $idx -ge $TOTAL ]] && break
    LAYER=${LAYERS[$idx]}
    echo "GPU $local_gpu -> layer=$LAYER"

    cat >> $TMPSCRIPT << EOF
CUDA_VISIBLE_DEVICES=$local_gpu python scripts/train_probe.py \\
    model=apertus \\
    training=apertus_stability_fp32 \\
    dataset=repetition_balanced_layer_sweep \\
    probe_config.layer=$LAYER \\
    'wandb_tags=[layer_$LAYER,sweep_${SWEEP_TIMESTAMP}]' \\
    > $REPO/cluster/log/sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_layer${LAYER}.log 2>&1 &
EOF
done

echo "wait" >> $TMPSCRIPT
echo 'echo "All probes in this array task done."' >> $TMPSCRIPT
chmod +x $TMPSCRIPT

srun --environment=$REPO/cluster/env.toml bash $TMPSCRIPT

rm -f $TMPSCRIPT
echo "Array task $SLURM_ARRAY_TASK_ID done."
