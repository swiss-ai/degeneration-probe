#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --job-name=degeneration-probe-archive
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Pack the unselected checkpoints of each listed run into one archive apiece:
#
#   sbatch cluster/archive_checkpoints.sbatch <file with one run directory per line>
#
# Pure file I/O against permanent storage, so it runs on a node rather than the
# login node, and each run is verified before anything of it is removed (see
# scripts/archive_checkpoints.py). A run that fails its check is reported and
# left untouched; the job carries on with the next one and exits non-zero at
# the end if any run failed.
set -uo pipefail
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
LIST=${1:?a file listing run directories is required}
cd $REPO
echo "Commit: $(git rev-parse HEAD)   List: $LIST ($(wc -l < "$LIST") runs)"
failed=0
while read -r run; do
    [[ -n $run ]] || continue
    printf '\n===== %s =====\n' "$run"
    if ! .venv/bin/python scripts/archive_checkpoints.py --run-dir "$run"; then
        echo "FAILED: $run" >&2
        failed=$((failed + 1))
    fi
done < "$LIST"
echo "archive job done, $failed failed"
exit $(( failed > 0 ))
