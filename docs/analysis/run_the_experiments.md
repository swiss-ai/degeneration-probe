# Prompt: run the degeneration-probe experiment programme

Paste everything below the line into a fresh Claude Code session started in
`/iopsstor/scratch/cscs/mdenegri/degeneration-probe`.

---

You are running a research programme end to end on a Slurm cluster. The plan is
already written and is not yours to redesign. Your job is to execute it in
order, verify each stage before moving on, and stop and ask when something does
not look right.

## The plan lives in the notebook

`notebooks/experiment_diary.py` is the source of truth. It is a marimo notebook
holding an experiment register as data: every run's exact Hydra overrides are
generated there, and every stage has a section explaining what it asks and what
to expect.

Read it first, in full. To read the register without a browser:

```bash
.venv/bin/python - <<'PY'
import re
from pathlib import Path
src = Path("notebooks/experiment_diary.py").read_text()
print(src[src.index("def _grid_runs"):src.index("BY_ID = ")])
PY
```

To print the exact commands for one stage, open the notebook and read the
`show_commands("S1")` cells, or reproduce the renderer: the plan is
`build_plan()` over `EXPERIMENTS`, and `render(slot)` turns a slot into an
`sbatch` line. **Generate commands from the register. Never hand-write
overrides**, because a run's identity is a hash of its full configuration and a
typo silently creates a second, different run instead of failing.

## Environment rules, which matter

- **Never run training, scoring or heavy evaluation on the login node.** Always
  `sbatch`. The login node is shared and killing it affects other people.
- Cheap things (`ls`, `git`, reading parquet, editing) are fine directly.
- Credentials come from `~/keys/.hf_token` and `~/keys/.wandb_key`.
- Never add a Claude co-author trailer to git commits.
- Job scripts already exist: `cluster/train.sbatch` (Hydra overrides pass
  straight through), `cluster/score_layers.sbatch <run_dir> "<depths>" <split>`,
  `cluster/baselines.sbatch`.
- **Walltime**: generous for training (a timeout throws away hours), tight for
  scoring (a timeout costs one cheap re-run, and short jobs backfill sooner).
  Training runs take about 80 minutes; ask for 3 hours. Scoring takes about
  3 minutes per depth plus 3 minutes of startup; ask for 40 minutes.

## Order of work

Stages are ordered by dependency. **Within a stage every run is independent and
should be queued at once.** Do not serialise what can run in parallel.

| stage | depends on | runs | what it decides |
|---|---|---|---|
| S0 baselines | nothing | 1 job | what a probe has to beat |
| S1 rule × window | nothing | 42 | the selection rule, the window, the depth |
| S2a horizon | nothing | 27 | whether the label horizon buys earliness |
| S2b soft labels | S1 | 12 | whether a graded target helps |
| S2c regression | S1 | 3 | whether a dense target helps |
| S2d class balance | S1 | 12 | whether the balance knobs matter |
| S3 adapted | S1, S2 | 6 | whether frozen features are the ceiling |
| S4 held-out test | S3 | scoring only | the one honest number |

**S0, S1 and S2a depend on nothing.** Start all three together. S2b, S2c and
S2d are listed as depending on S1 only because they should use the window S1
selects; if you prefer, run them at the register's default window and note it.

## After every stage, verify before continuing

1. **Every job finished.** `sacct -u $USER -S <date> --parsable2` and check for
   anything that is not `COMPLETED`. A `TIMEOUT` or `FAILED` run must be
   re-queued, not quietly skipped.
2. **Every run wrote what it should.** For each run directory: `run_info.json`
   with `"status": "finished"`, a `history.parquet`, and checkpoints.
3. **No run hit the step cap.** `max_steps` is 2000 and is meant to be a cap
   the stopping rule never reaches. If runs are stopping at 2000, the budget is
   too small and the comparison is not yet valid. Report this rather than
   proceeding.
4. **No probe collapsed.** A score spread near zero means the probe emits a
   constant, which converges nicely and distinguishes nothing.
5. **The realized token budget matches the requested one** for every run. It is
   recorded per run; a mismatch means one recipe got more gradient than the
   others and the comparison inside that stage is void.

Then score, then read the stage's section in the notebook and report what the
tables say.

## Scoring, which is a separate step

Training writes no evaluation. Scoring runs afterwards from a checkpoint.

**Always score every depth, on validation, for every run.** One job per run:

```bash
for run in outputs/*/2*/; do
    sbatch --time=03:00:00 cluster/score_layers.sbatch \
        "$(realpath $run)" "$(seq -s' ' 1 31)" val
done
```

A depth costs about three minutes and a run is therefore about an hour and a
half. Scoring a chosen few depths would be cheaper and is deliberately not done:
it would require deciding which depths a result could be read at before the
results exist, and it would make one recipe's depth profile an assumption
carried over to the others rather than something measured.

Do not score the test splits at any stage before S4.

## Reporting

After each stage, report: how many runs finished, anything that failed and what
you did about it, and what the stage's tables say against what its notebook
section said to expect. Be specific with numbers. If a result contradicts the
expectation written in the notebook, say so plainly rather than smoothing it
over: several expectations in this programme have already turned out wrong, and
finding that out is the point.

Do not update the notebook's narrative with results unless asked.

## Things that have already gone wrong here, so watch for them

- **A window too small to express the label.** A centred window shows half its
  width of run-up, so a horizon or decay length larger than that trains
  identical data at two different settings and reports them as two points
  differing in nothing. The register already respects this; do not "simplify" a
  window down.
- **A metric that cannot separate what it selects.** Rollout-level recall is
  saturated here. If you find yourself comparing recipes on a number that takes
  three distinct values, you are reading the wrong view. Coverage and distance
  from the frontier are the ones that move.
- **A silently doubled budget.** A step cannot be smaller than one micro-batch,
  so a large window with a small budget quietly gets more tokens. There is now
  a guard that raises; if you see it, lower the batch size rather than raising
  the budget for one recipe only.
- **Monitor numbers are not results.** Anything from `metrics.jsonl` comes from
  a thinned split and exists to steer the run. Report only what comes from the
  scoring pipeline.
- **Single seeds cannot rank.** Three seeds are required before saying one
  recipe beats another. One seed is a pilot.

## The protocol document

`docs/analysis/probe_eval_and_training_protocol.md` is the full design: the
corpus, the label families, the selection ladder, the four evaluation views, and
what is tuned where. Read it if a decision is not obvious from the notebook. It
is kept current; if you find it disagreeing with the code, that is a bug in one
of them and worth reporting.

## When you are done

All stages complete, every run accounted for, and a summary of what each stage
decided. S4 is scored once and never repeated.
