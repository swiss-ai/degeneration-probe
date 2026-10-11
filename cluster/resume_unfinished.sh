#!/bin/bash
# Resubmit every run of the frozen sweep that has not finished (timed out, failed,
# never started). A relaunched line finds its unfinished attempt and resumes from
# the last checkpoint, so this is safe to repeat. It refuses to run while any
# training job is queued, since a second job on the same run would write into the
# same attempt directory.
#
#   ./cluster/resume_unfinished.sh            submit
#   DRY_RUN=1 ./cluster/resume_unfinished.sh  list only
set -euo pipefail
cd /iopsstor/scratch/cscs/mdenegri/degeneration-probe
LAUNCH=${LAUNCH:-cluster/launch_frozen_sweep.sh}
DRY_RUN=${DRY_RUN:-0}

if [[ $DRY_RUN == 0 ]] && squeue -u "$USER" -h -n degeneration-probe-train -o "%i" | grep -q .; then
    echo "training jobs are still queued or running; wait for them before resuming" >&2
    exit 1
fi

.venv/bin/python - "$DRY_RUN" "$LAUNCH" <<'PY'
import shlex, subprocess, sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from degeneration_probe.config import ExperimentConfig
from degeneration_probe.training.run_identity import derive_run_name
import json
dry = sys.argv[1] != "0"
lines = [l.rstrip() for l in open(sys.argv[2]) if l.startswith("sbatch")]
todo = []
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    for line in lines:
        overrides = shlex.split(line)[3:]
        cfg = ExperimentConfig.from_dict(OmegaConf.to_container(compose("main", overrides=overrides), resolve=True))
        root = Path(cfg.training.checkpoint.root_dir) / derive_run_name(cfg)
        finished = any(
            json.loads(p.read_text()).get("status") == "finished"
            for p in root.glob("2*/run_info.json") if p.is_file()
        ) if root.is_dir() else False
        if not finished:
            todo.append((root.name, line))
print(f"{len(todo)} of {len(lines)} runs not finished")
for name, line in todo:
    print(("would submit " if dry else "submit ") + name)
    if not dry:
        subprocess.run(line, shell=True, check=True)
PY
