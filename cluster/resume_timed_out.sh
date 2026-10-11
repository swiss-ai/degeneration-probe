#!/bin/bash
# Resume the training jobs that ran out of walltime or lost their node, and
# rebuild the chain behind each one.
#
#   ./cluster/resume_timed_out.sh [since]            e.g. 2026-10-11T15:00
#   DRY_RUN=1 ./cluster/resume_timed_out.sh [since]
#
# For every training job since <since> that ended in TIMEOUT or NODE_FAIL:
#   - the run it was training is read from its log, and skipped when that run
#     has finished since or another queued training job already carries it;
#   - the jobs that depended on it (now never satisfiable) are cancelled;
#   - its original launch line is resubmitted, which resumes from the run's
#     last checkpoint, and the scoring is queued behind it again: an adapted
#     run scores straight after training, a frozen run replays first.
#
# Launch lines are matched to runs by composing each line's configuration, the
# same way the trainer names a run, and cached in outputs/analysis.
set -euo pipefail
cd /iopsstor/scratch/cscs/mdenegri/degeneration-probe
SINCE=${1:-2026-10-10T19:40}
DRY_RUN=${DRY_RUN:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/outputs}
LAUNCHES="cluster/launch_frozen_sweep.sh cluster/launch_frozen_transfer.sh cluster/launch_lora_depth.sh"
CACHE=$OUTPUT_ROOT/analysis/launch_lines_by_run.json
LOG=cluster/log

# Map every launch line to the run name it produces, reusing earlier answers.
.venv/bin/python - "$CACHE" $LAUNCHES <<'PY'
import json, shlex, sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from degeneration_probe.config import ExperimentConfig
from degeneration_probe.training.run_identity import derive_run_name
cache_path = Path(sys.argv[1]); files = sys.argv[2:]
cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
by_line = {entry["line"]: name for name, entry in cache.items()}
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    for file in files:
        for line in open(file):
            line = line.rstrip()
            if not line.startswith("sbatch") or line in by_line:
                continue
            overrides = shlex.split(line)[3:]
            cfg = ExperimentConfig.from_dict(OmegaConf.to_container(compose("main", overrides=overrides), resolve=True))
            name = derive_run_name(cfg)
            cache[name] = {"line": line, "regime": cfg.training.features.regime, "file": file}
            by_line[line] = name
cache_path.parent.mkdir(parents=True, exist_ok=True)
cache_path.write_text(json.dumps(cache, indent=1))
print(f"{len(cache)} launch lines known")
PY

# Runs already carried by a queued training job, so nothing is resumed twice.
declare -A QUEUED
for j in $(squeue -u "$USER" -h -n degeneration-probe-train -o %i); do
    name=$(grep -m1 "^Run name:" "$LOG/degeneration-probe-train-$j.log" 2>/dev/null | awk '{print $3}' || true)
    [[ -n $name ]] && QUEUED[$name]=$j
done

resumed=0
while read -r job state; do
    [[ -n $job ]] || continue
    log=$LOG/degeneration-probe-train-$job.log
    name=$(grep -m1 "^Run name:" "$log" 2>/dev/null | awk '{print $3}' || true)
    run_dir=$(grep -m1 "^Run directory:" "$log" 2>/dev/null | awk '{print $3}' || true)
    if [[ -z $name || -z $run_dir ]]; then
        echo "skip   $job ($state): no run name in its log"; continue
    fi
    status=$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" "$run_dir/run_info.json" 2>/dev/null || echo unknown)
    if [[ $status == finished ]]; then
        echo "done   $job $name (finished since)"; continue
    fi
    if [[ -n ${QUEUED[$name]:-} ]]; then
        echo "queued $job $name (already carried by job ${QUEUED[$name]})"; continue
    fi
    read -r line regime < <(.venv/bin/python -c "
import json, sys
cache = json.load(open(sys.argv[1])); entry = cache.get(sys.argv[2])
print(json.dumps(entry['line']) if entry else '', entry['regime'] if entry else '')" "$CACHE" "$name")
    if [[ -z $line ]]; then
        echo "skip   $job $name: no launch line produces this run"; continue
    fi
    line=$(.venv/bin/python -c "import json,sys; print(json.loads(sys.argv[1]))" "$line")
    dependents=$(squeue -u "$USER" -h -o "%i %E" | awk -v j="$job" '$2 ~ ("afterok:" j "\\(") {print $1}' | tr '\n' ' ')
    echo "resume $job $name ($state, status $status); cancelling dependents: ${dependents:-none}"
    if [[ $DRY_RUN == 1 ]]; then
        continue
    fi
    [[ -n $dependents ]] && scancel $dependents
    new=$(eval "${line/sbatch /sbatch --parsable }")
    if [[ $regime == adapted ]]; then
        score=$(sbatch --parsable --dependency=afterok:$new cluster/score_adapted_selected.sbatch "$new")
        echo "  train $new -> score-adapted $score" | tee -a "$OUTPUT_ROOT/chained_submissions.txt"
    else
        replay=$(sbatch --parsable --dependency=afterok:$new cluster/replay.sbatch "$run_dir" 100)
        score=$(SBATCH_TIMELIMIT=12:00:00 sbatch --parsable --dependency=afterok:$replay cluster/score_selected_all.sbatch "$run_dir")
        echo "  train $new -> replay $replay -> score $score" | tee -a "$OUTPUT_ROOT/chained_submissions.txt"
    fi
    QUEUED[$name]=$new
    resumed=$((resumed + 1))
done < <(sacct -u "$USER" -S "$SINCE" -X -n -o JobID,JobName%30,State | awk '$2 ~ /train/ && ($3=="TIMEOUT" || $3=="NODE_FAIL") {print $1, $3}')
echo "$resumed resumed"
