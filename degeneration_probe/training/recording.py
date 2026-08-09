"""Everything a finished run leaves behind on disk.

A run directory has to answer three questions long after the job is gone:
what was trained, on what data, and how it behaved while training. The
tracking UI answers them while the sweep is live, but it is not a data source
for later analysis, so every number sent to the tracker is mirrored locally in
the same shape. The metric history in particular is written from the same hook
the tracker listens on, which is what keeps the two identical by construction
rather than by convention.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import TrainerCallback

METRICS_FILE = "metrics.jsonl"
HISTORY_FILE = "history.parquet"
RUN_INFO_FILE = "run_info.json"
DATASET_SUMMARY_FILE = "dataset_summary.json"
RESOLVED_CONFIG_FILE = "resolved_config.json"
FINAL_EVALUATION_FILE = "final_evaluation.json"
FINAL_WEIGHTS_DIR = "final"


def _git(*args: str) -> Optional[str]:
    try:
        repo = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "wandb", "pandas"):
        try:
            versions[package] = __import__(package).__version__
        except Exception:
            versions[package] = None
    return versions


def environment_info() -> Dict[str, Any]:
    """Where and from which code state this run executed."""
    import torch

    devices = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "devices": devices,
        "versions": _package_versions(),
    }


ATTEMPT_FORMAT = "%Y%m%dT%H%M%S"
ATTEMPT_PATTERN = re.compile(r"^\d{8}T\d{6}(-\d+)?$")
LATEST_LINK = "latest"


@dataclass
class RunLocation:
    """Where one attempt at a run writes, and what it carries over."""

    run_dir: Path
    run_name: str
    attempt: str
    resume_from: Optional[str] = None
    wandb_id: Optional[str] = None
    continued: bool = False


def _attempt_dirs(run_root: Path) -> List[Path]:
    if not run_root.is_dir():
        return []
    return sorted(
        path for path in run_root.iterdir() if path.is_dir() and ATTEMPT_PATTERN.match(path.name)
    )


def _point_latest_at(run_root: Path, attempt: str) -> None:
    """Keep a stable path to the newest attempt, replaced atomically."""
    link = run_root / LATEST_LINK
    staged = run_root / f".{LATEST_LINK}.staged"
    try:
        if staged.is_symlink() or staged.exists():
            staged.unlink()
        staged.symlink_to(attempt, target_is_directory=True)
        os.replace(staged, link)
    except OSError:
        # A filesystem without symlinks is not a reason to lose a run.
        pass


def prepare_run_location(root: Path, run_name: str, resume: str) -> RunLocation:
    """Give this attempt its own directory under the run's parent.

    Runs of one configuration share a parent named after the configuration and
    differ only by a timestamped subdirectory, so repeating a run never
    overwrites the one before it and the attempts of a recipe stay together.
    An interrupted attempt is continued in place rather than forked, so its
    metric history and its tracker run stay in one piece.
    """
    from transformers.trainer_utils import get_last_checkpoint

    run_root = Path(root) / run_name
    if resume == "auto":
        for previous in reversed(_attempt_dirs(run_root)):
            info = {}
            info_path = previous / RUN_INFO_FILE
            if info_path.is_file():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                except ValueError:
                    info = {}
            if info.get("status") == "finished":
                break
            checkpoint = get_last_checkpoint(str(previous))
            if checkpoint:
                _point_latest_at(run_root, previous.name)
                return RunLocation(
                    run_dir=previous,
                    run_name=run_name,
                    attempt=previous.name,
                    resume_from=checkpoint,
                    wandb_id=(info.get("wandb") or {}).get("id"),
                    continued=True,
                )
            break

    stamp = time.strftime(ATTEMPT_FORMAT, time.localtime())
    # Two jobs of one configuration can start within the same second, and the
    # loser of that race must still get a directory of its own.
    for suffix in range(100):
        attempt = stamp if suffix == 0 else f"{stamp}-{suffix}"
        run_dir = run_root / attempt
        try:
            run_dir.mkdir(parents=True)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(f"Could not claim an attempt directory under {run_root}")
    _point_latest_at(run_root, attempt)
    return RunLocation(
        run_dir=run_dir,
        run_name=run_name,
        attempt=attempt,
        resume_from=None if resume in {"auto", "never"} else resume,
    )


@dataclass
class RunInfo:
    """The identity card of a run, rewritten as the run progresses."""

    run_name: str
    group: str
    tags: List[str]
    axes: Dict[str, Any]
    run_dir: str
    attempt: str
    environment: Dict[str, Any] = field(default_factory=environment_info)
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    wandb: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def write(self, run_dir: Path) -> None:
        path = Path(run_dir) / RUN_INFO_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in self.__dict__.items()}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def finish(self, run_dir: Path, *, status: str, error: Optional[str] = None) -> None:
        self.status = status
        self.error = error
        self.finished_at = time.time()
        self.duration_seconds = self.finished_at - self.started_at
        self.write(run_dir)


class ResampleWindows(TrainerCallback):
    """Redraw the training windows at the start of each epoch.

    Free augmentation, and it makes a comparison between placement rules a
    comparison over the pool rather than over one arbitrary draw.
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        if epoch:
            self.dataset.resample(epoch)


class CollapseGuard(TrainerCallback):
    """Stop a run whose probe has stopped distinguishing anything.

    A probe can minimize its loss by ignoring its input and emitting one
    constant value, which under a strong class weight is a real attractor. Such
    a run posts a plausible loss, and rank metrics over constant scores are
    undefined or arbitrary tie-breaking, so it would enter a comparison table
    reading as a weak recipe rather than as a broken one.
    """

    def __init__(self, threshold: float = 0.01, key: str = "val/prediction_std") -> None:
        self.threshold = float(threshold)
        self.key = key

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.key not in metrics:
            return
        spread = float(metrics[self.key])
        if spread < self.threshold:
            raise RuntimeError(
                f"the probe collapsed: {self.key}={spread:.6g} is below {self.threshold:g} at "
                f"step {state.global_step}. It is emitting one value for every token, which "
                "converges and scores plausibly while distinguishing nothing."
            )


class RunRecorder(TrainerCallback):
    """Mirror every logged record into the run directory as it is produced.

    The callback fires on the same hook the tracker's own callback uses, so the
    local history and the remote history cannot drift apart. Records are
    appended line by line, which means an interrupted run still leaves a
    complete history up to the moment it stopped.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / METRICS_FILE
        self._started = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            "wall_time": time.time() - self._started,
            **{key: value for key, value in logs.items() if key != "epoch"},
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        self.write_history()

    def write_history(self) -> Optional[Path]:
        """Collapse the append-only log into one table per step."""
        if not self.metrics_path.is_file():
            return None
        import pandas as pd

        records = [
            json.loads(line)
            for line in self.metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            return None
        frame = pd.DataFrame(records)
        # Training and validation records arrive separately at the same step;
        # one row per step is what a plot wants.
        numeric = frame.select_dtypes(include="number").columns
        history = frame.groupby("step", as_index=False)[list(numeric)].mean()
        path = self.run_dir / HISTORY_FILE
        history.to_parquet(path, index=False)
        return path
