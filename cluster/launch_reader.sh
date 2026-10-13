#!/bin/bash
# The text-only baseline: Qwen2.5-1.5B reads the prompt and the generated text
# (re-tokenized, scripts/build_reader_dataset.py) and is trained with the selected
# recipe, rank-16 LoRA on every block and the probe at its last block, the same
# step budget and cadence as the adapted probes. Each run is then scored at the
# checkpoint the selection rule picks and put back on the generator's tokens.
#
#   ./cluster/launch_reader.sh <dataset> <reader build job id>
set -euo pipefail
cd /iopsstor/scratch/cscs/mdenegri/degeneration-probe
DATASET=${1:?a generator dataset is required}
AFTER=${2:-}
DEP=${AFTER:+--dependency=afterok:$AFTER}
for SEED in 42 43 44; do
    TRAIN=$(sbatch --parsable --time=12:00:00 $DEP cluster/train.sbatch \
        model=qwen2p5-1p5b dataset=degeneration-dataset-$DATASET-reader-qwen2p5-1p5b \
        training.label.family=frontier_hard training.label.horizon=256 training.loss.bce.use_pos_weight=true \
        training.loss.name=bce training.selection.anchor=centered training.selection.positive_fraction=0.25 \
        training.selection.strategy=frontier_window training.selection.window_size=512 \
        training.features.regime=adapted training.lora.enabled=true training.lora.layers=all training.lora.rank=16 \
        training.optimizer.lora_learning_rate=1e-5 training.optimizer.probe_learning_rate=1e-4 \
        training.probe.layers=null training.probe.layer=27 training.budget.tokens_per_step=4096 \
        training.runtime.per_device_train_batch_size=1 training.runtime.max_steps=3000 \
        training.validation.strategy=steps training.validation.steps=100 training.validation.max_rollouts=null \
        'training.validation.final_splits=[]' training.checkpoint.strategy=steps training.checkpoint.steps=100 \
        training.checkpoint.save_total_limit=null \
        training.checkpoint.root_dir=/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/outputs \
        training.stopping.enabled=false 'training.wandb.tags=["exp:reader-text-only"]' training.runtime.seed=$SEED)
    SCORE=$(sbatch --parsable --dependency=afterok:$TRAIN cluster/score_adapted_selected.sbatch $TRAIN)
    REMAP=$(sbatch --parsable --dependency=afterok:$SCORE cluster/reader_remap.sbatch $TRAIN $DATASET)
    echo "$DATASET seed $SEED: train $TRAIN, score $SCORE, remap $REMAP"
done
