"""Applying the checkpoint rule to a run as it trains.

A run trains a head at every depth and saves all of them on the validation
cadence. This watches the record each depth produces at each of those steps,
decides where a depth has stopped improving, and ends the run once its last
depth has stopped.

The decision is not made here. It is made by ``head_selection``, which the
replay of saved checkpoints also calls, so a run that stops itself and a run
that is judged afterwards are judged by one rule rather than by two that
resemble each other. What this adds is only the bookkeeping: turning a stream
of validation records into the table the rule reads, and writing that table
next to the checkpoints it describes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from transformers import TrainerCallback

from degeneration_probe.evaluation.head_selection import apply_rule_to_run
from degeneration_probe.evaluation.protocol import WARNING_BANDS

SELECTION_HISTORY_FILE = "selection_history.parquet"
SELECTION_OUTCOMES_FILE = "selection_outcomes.json"

# The columns the rule reads, in the shape the replay writes them, so one
# analysis reads either source without knowing which produced it.
RECORD_FIELDS = (
    "budget_tau",
    "budget_realized_fpr",
    "recall_at_budget",
    "in_pattern_recall",
    "never_fired_positives",
    "median_offset",
    *(f"warning_recall_{band}" for band in WARNING_BANDS),
)

_LAYER_KEY = re.compile(r"^layer(\d+)$")


def records_from_metrics(
    metrics: Dict[str, float], *, prefix: str, layers: Sequence[int]
) -> List[Dict[str, float]]:
    """One row per depth, pulled out of a validation metrics dictionary.

    A single-depth run reports its record unprefixed and a multi-depth run
    reports one namespace per depth, so the caller says which depths it trained
    and this finds each of them wherever it was written.
    """
    rows = []
    for layer in layers:
        namespaced = f"{prefix}/layer{layer:02d}/"
        plain = f"{prefix}/"
        row = {"layer": int(layer)}
        for field in RECORD_FIELDS:
            if namespaced + field in metrics:
                row[field] = float(metrics[namespaced + field])
            elif len(layers) == 1 and plain + field in metrics:
                row[field] = float(metrics[plain + field])
        # A record needs the objective and the gate to be decidable at all; a
        # step that produced neither says nothing about where a depth is.
        if len(row) > 1:
            rows.append(row)
    return rows


class HeadSelection(TrainerCallback):
    """Record what each depth is doing, and stop the run when none is moving.

    A depth is watched on whichever quantity it still has to earn and stops when
    that quantity has been flat for the configured number of evaluations. The
    run ends when every depth has stopped, which means its cost is set by its
    slowest depth rather than its typical one, and a depth that never learns to
    see the loop stops on the same terms as one that did.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        rule,
        layers: Sequence[int],
        prefix: str = "val",
        stop_when_finished: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.rule = rule
        self.layers = [int(layer) for layer in layers]
        self.prefix = prefix
        self.stop_when_finished = stop_when_finished
        self.history: List[Dict[str, float]] = []
        self.outcomes = None

    # ---- the trajectory ----

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or not state.is_world_process_zero:
            return
        rows = records_from_metrics(metrics, prefix=self.prefix, layers=self.layers)
        if not rows:
            return
        step = int(state.global_step)
        # Resuming re-evaluates the step it resumed from, and a duplicated step
        # would give that evaluation two votes in the patience count.
        self.history = [row for row in self.history if row["step"] != step]
        self.history.extend({"step": step, **row} for row in rows)
        self.write()
        if self.outcomes is None or self.outcomes.empty:
            return
        finished = self.outcomes["stopped_at"].notna().all()
        if finished and self.stop_when_finished:
            selectable = int(self.outcomes["selected_step"].notna().sum())
            print(
                f"Every depth has stopped improving by step {step}: "
                f"{selectable} of {len(self.outcomes)} are selectable. Ending the run."
            )
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.write()

    # ---- what it leaves behind ----

    def write(self) -> Optional[Path]:
        """The trajectory and the rule's verdict, rewritten as the run goes.

        Written every time rather than once at the end, so a run killed by the
        walltime still leaves a table its depths can be judged from.
        """
        if not self.history:
            return None
        import pandas as pd

        frame = pd.DataFrame(self.history).sort_values(["layer", "step"])
        frame = frame.reset_index(drop=True)
        path = self.run_dir / SELECTION_HISTORY_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        try:
            self.outcomes = apply_rule_to_run(frame, self.rule)
        except KeyError:
            # The objective or the gate was never measured, which happens when
            # the monitoring split holds no degenerate answers. Nothing to
            # decide, and nothing to stop on.
            self.outcomes = None
            return path
        (self.run_dir / SELECTION_OUTCOMES_FILE).write_text(
            json.dumps(
                {
                    "rule": {
                        "floor": self.rule.floor,
                        "band": self.rule.band,
                        "tolerance": self.rule.tolerance,
                        "patience": self.rule.patience,
                        "objective": self.rule.objective,
                    },
                    # A depth that never became selectable has no step and no
                    # value. Pandas spells that absence as a float NaN, which
                    # ``json`` writes as a bare NaN literal that a strict reader
                    # refuses, so it is spelled as null instead.
                    "depths": self.outcomes.astype(object)
                    .where(self.outcomes.notna(), None)
                    .to_dict("records"),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return path
