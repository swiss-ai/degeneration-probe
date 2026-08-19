# Running the adapted stage

This is a step-by-step guide to launching 36 training runs on your allocation.
The runs write into a shared output directory, so the probes of the programme
stay together and a result can be read later without knowing whose allocation
paid for it.

Everything you need is in the repository. One script launches all 36 runs; the
rest of this document is the checks worth doing first and the way to tell that a
run is behaving.

---

## What you are running

A probe is a linear head on one layer of a frozen language model, trained to say
how close a generation is to collapsing into a repetition loop. This stage asks
whether letting adapters move the representation underneath that head buys
anything, so each run trains LoRA adapters alongside the head.

Twelve recipes, three seeds each. The recipes are not arbitrary: each is one that
an earlier stage selected, and each is trained at the depth its frozen
counterpart scored best at. Nothing about them needs to be chosen or tuned.

A run ends itself. It measures, every fifty steps, how much of the run-up to a
repetition loop it flags, and stops once that has not improved for four
consecutive measurements. Expect most runs to end between step 200 and step 800
rather than at the 2000-step cap.

---

## Before you launch

### 1. Group membership

The dataset and the output directory are shared through the `infra01` group.

```bash
id -Gn | tr ' ' '\n' | grep -qx infra01 && echo "ok" || echo "MISSING: ask for infra01"
```

### 2. The repository

The job scripts derive their paths from your username, so the clone has to sit at
exactly this path:

```bash
git clone git@github.com:Luca-Sartori/degeneration-probe.git \
    /iopsstor/scratch/cscs/$USER/degeneration-probe
cd /iopsstor/scratch/cscs/$USER/degeneration-probe
git checkout dataset-v2-pilot
git pull
```

If you already have it, `git pull` is enough. You need the branch
`dataset-v2-pilot`.

### 3. Credentials

Two files are read by the job script. Check both are present and non-empty:

```bash
for f in ~/keys/.hf_token ~/keys/.wandb_key; do
    [[ -s $f ]] && echo "ok      $f" || echo "MISSING $f"
done
```

`~/keys/.hf_token` is required. `~/keys/.wandb_key` is optional and falls back to
the credentials in `~/.netrc`; without either, tracking is the only thing that
fails, but it fails late, so it is worth checking now. Runs log to whichever
Weights & Biases account those credentials belong to, which is fine: nothing
downstream reads the tracker, and every number a run produces is also written
into its own directory.

### 4. The container

The image supplies the Python environment. The code itself is read from your
clone at run time, so the image never needs rebuilding when the code changes.

```bash
ls -l $SCRATCH/ce-images/degeneration-probe+25.06-dataset-v2-pilot.sqsh
```

If it is missing, or if a job fails at container start, copy a known-good one:

```bash
mkdir -p $SCRATCH/ce-images
cp /iopsstor/scratch/cscs/mdenegri/ce-images/degeneration-probe+25.06-dataset-v2-pilot.sqsh \
   $SCRATCH/ce-images/
```

It is 23 GB, so the copy takes a few minutes.

### 5. Write access to the output root

The runs go here:

```
/iopsstor/scratch/cscs/mdenegri/degeneration-probe/outputs
```

Check that you can write to it:

```bash
test -w /iopsstor/scratch/cscs/mdenegri/degeneration-probe/outputs \
    && echo "ok" || echo "NOT WRITABLE: see below"
```

If it is not writable, its owner grants group access with:

```bash
setfacl -m g::rwx /iopsstor/scratch/cscs/mdenegri/degeneration-probe/outputs
```

`chmod g+w` does not work here. The directory carries an access control list, and
`chmod` only widens its mask while leaving the group entry read-only. The
directory already inherits group-write onto everything created inside it, so once
this is set your runs stay readable and writable by both accounts.

### 6. Nothing else is running against the same directory

Two jobs training the same recipe at the same time will resume into the same run
directory and overwrite each other. Before you launch, the other account cancels
its own queued training jobs. Confirm the queue is clear of them:

```bash
squeue -A infra01 -o "%.10i %.8u %.30j %.8T" | grep degeneration-probe-train
```

Only your own jobs should appear once you have launched.

---

## Launch

```bash
cd /iopsstor/scratch/cscs/$USER/degeneration-probe
./cluster/relaunch_adapted.sh
```

That submits 36 jobs and prints where they will write. The script refuses to run
if the output directory is not writable, so a permissions problem stops it before
anything is queued rather than after.

Walltimes vary from 7 to 12 hours per run, because holding the token budget fixed
makes a narrow window run many more answers through the model per step than a
wide one. Most runs stop long before their limit.

To send the runs somewhere else instead, for a trial:

```bash
OUTPUT_ROOT=$SCRATCH/degeneration-probe/outputs ./cluster/relaunch_adapted.sh
```

---

## Checking that a run is behaving

Roughly twenty minutes in, a run has written its first measurements. Pick any run
directory and look:

```bash
cd /iopsstor/scratch/cscs/mdenegri/degeneration-probe/outputs
ls -td *lora-all*/2* | head -1
```

Inside it, `selection_history.parquet` holds one row per measurement and
`selection_outcomes.json` holds the verdict so far. A healthy run looks like this:

```
step   in_pattern_recall   warning_recall_256
 50          0.39                0.0062
100          0.57                0.0273
150          0.58                0.0397     <- best so far
200          0.50                0.0342
```

Three things are worth confirming on the first run that gets going:

- **The startup log says it is monitoring 3634 answers.** Grep the job log for
  `Monitoring on`. If it reports a few hundred instead, the run is measuring
  against a threshold resting on three healthy answers and its numbers will move
  with the sample rather than with the probe.
- **`in_pattern_recall` climbs past 0.3 within the first few measurements.** Below
  that a probe has not yet learned to see a loop it is already inside, and its
  checkpoints do not qualify for selection.
- **The run stops on its own.** `selection_outcomes.json` records `stopped_at`
  when it does. A run that reaches step 2000 hit the backstop instead, which is
  worth mentioning rather than ignoring.

---

## What must not be changed

The launcher already sets all of this. It matters because each run is compared
against a frozen counterpart trained under the same conditions, and a difference
in any of these turns that comparison into a comparison of something else.

| | |
|---|---|
| Monitoring split | the whole validation split, never a subsample |
| Measurement and checkpoint cadence | every 50 steps, all checkpoints kept |
| Stopping rule | floor 0.3, band 256, tolerance 0.002, patience 4 |
| Step cap | 2000, as a backstop |
| Tokens per optimizer step | 4096, with one answer per micro-batch |
| Adapters | all layers, rank 16, learning rate 1e-5 |
| Probe learning rate | 1e-4 |
| Seeds | 42, 43, 44 |

One thing to know when reading results: the checkpoint a run is judged on is the
one named in `selection_outcomes.json`. The weights left in `final/` are a
convenience and are not always the same checkpoint.

---

## What is deliberately not launched

One recipe is missing from the 36 by design: the hard-negative window rule, which
would be three more runs.

Placing a hard negative window requires knowing where a healthy answer looks
repetitive. That signal reaches the training data when the model is frozen and
does not reach it when adapters are training, so those runs fail at startup with:

```
ValueError: frontier_window_hard_negative needs a per-token repetition signal
for every negative rollout; none was supplied for record 2
```

Do not try to run them. It is a gap in the code rather than a configuration
problem, and it is commented out in the launcher with a note. The stage was
planned so that its conclusions survive without this recipe.

---

## If a run is interrupted

Runs resume. Submitting the same command again continues an unfinished attempt
from its last checkpoint, in the same directory, with its history intact:

```bash
grep -n "<the recipe you want>" cluster/relaunch_adapted.sh
```

and resubmit that line. A run that already finished is never resumed; a repeat
starts a fresh attempt beside it, which is harmless.

---

## Scoring the checkpoints, once the runs are done

Training measures just enough to decide which of a run's checkpoints is worth
keeping. The reported results need something more expensive: one score for every
token of every answer, put through the four views. That costs a pass over the
data, so it is spent once per run, on the checkpoint the rule named.

Two things follow. The first is that this step is not optional: nothing in a
finished run directory yet contains the numbers a result is read from. The second
is that the checkpoint to score is the one recorded in `selection_outcomes.json`,
never the weights left in `final/`. Those are wherever training happened to stop,
which is typically a few hundred steps past the peak and a third of its value.

### Wait until the runs have finished

```bash
squeue -u $USER -n degeneration-probe-train
```

Scoring a run that is still training reads a half-written checkpoint. There is no
harm in scoring some runs while others are still going, as long as each run
itself has finished.

### Submit the scoring jobs

```bash
cd /iopsstor/scratch/cscs/$USER/degeneration-probe
DRY_RUN=1 ./cluster/score_selected.sh    # see what it would do
./cluster/score_selected.sh              # do it
```

It reads each run's own verdict, skips anything unfinished, anything already
scored, and any run where no checkpoint ever qualified, and submits one job per
run. Each takes about fifteen minutes.

The dry run prints one line per run so the checkpoints can be eyeballed before
anything is queued:

```
would  ..._frontier256_hard_bce_lora-all_s42_cc037d07  checkpoint-150  [val test_indomain]
would  ..._frontier256_hard_bce_lora-all_s43_ed05b892  checkpoint-200  [val test_indomain]
```

Two splits are scored, not three. The held-out domains are read once, at the very
end, for the probe that is actually reported; scoring them for all of these would
spend the one measurement that is meant to stay untouched. Override with
`SPLITS="val test_indomain test_heldout_domains"` only if that is the intention.

### Turn the scores into the four views

This part is arithmetic over a table of stored scores. It needs no GPU and takes
a couple of minutes for the whole batch, but it does need a Python environment
with the project installed, which the container supplies to jobs and the login
node does not. If you do not already have one:

```bash
cd /iopsstor/scratch/cscs/$USER/degeneration-probe
uv venv && uv pip install -e .
```

Then:

```bash
for scores in /iopsstor/scratch/cscs/mdenegri/degeneration-probe/outputs/*/2*/scores; do
    .venv/bin/python scripts/evaluate_scores.py --run-dir "${scores%/scores}"
done
```

Thresholds are chosen on the validation split alone, written to
`decision_thresholds.json`, and read back unchanged for every other split, so
validation has to be scored before any test split can be reported.

This step is also fine to leave alone. The scores are already in the owner's
scratch by then, and turning them into views is a two-minute job on a machine
that already has the environment. The part that needs your allocation is the
scoring above.

### Where everything lands

Inside each run's own directory, which is to say wherever the run itself is:

```
outputs/<run_name>/<attempt>/
  selection_outcomes.json      which checkpoint was chosen, and why
  scoring_provenance.json      which weights produced the scores beside it
  scores/<split>.parquet       one score per token per answer
  decision_thresholds.json     frozen on validation, reused on every split
  evaluation/<split>/*.csv     the four views
```

Nothing needs moving afterwards. A directory of scores is read as the output of
one scorer, so scoring a second checkpoint into the same place is refused rather
than silently mixed; `--output-dir` gives a second checkpoint its own place if
two are ever wanted side by side.

## When they are done

Nothing further is needed from you. Each run leaves its measurements, its
verdict, and every checkpoint in its own directory, and the analysis reads them
from there.

A quick count of how far the batch has got:

```bash
cd /iopsstor/scratch/cscs/mdenegri/degeneration-probe/outputs
printf '%-10s %-6s %-9s %s\n' STATUS STEP SELECTED RUN
for verdict in */2*/selection_outcomes.json; do
    run=${verdict%/*}
    printf '%-10s %-6s %-9s %s\n' \
        "$(jq -r '.status' "$run/run_info.json")" \
        "$(jq -r '.training.global_step // "-"' "$run/run_info.json")" \
        "$(jq -r '.depths[0].selected_step // "-"' "$verdict")" \
        "$(basename "$(dirname "$run")")"
done
```

`SELECTED` is the step whose checkpoint that run is judged on. A dash there on a
finished run means no checkpoint ever qualified, which is a result worth
reporting rather than a failure to fix.
