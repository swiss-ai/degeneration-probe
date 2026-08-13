"""Rename run directories that were named after a depth they never committed to.

A run carrying a head at every depth left the single-layer setting at whatever
it happened to be and was named after it, so its directory, its group and its
layer axis all claim one depth while the run trained many.

Only the depth segment of the name is rewritten. The fingerprint is left alone:
it records the settings as they were hashed at the time, and recomputing it
today would drift with every field the config has gained since, which would
break the link between a directory, its `config:` tag and its tracker run.

    python rename_multihead_runs.py            # dry run
    python rename_multihead_runs.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path("/iopsstor/scratch/cscs/mdenegri/degeneration-probe")
sys.path.insert(0, str(REPO))

from degeneration_probe.config import ExperimentConfig  # noqa: E402
from degeneration_probe.training.run_identity import depth_label  # noqa: E402

OUTPUTS = REPO / "outputs"


def planned_renames():
    """Every run directory whose name names one depth while reading many."""
    plan = []
    for run_dir in sorted(p for p in OUTPUTS.iterdir() if p.is_dir()):
        attempts = [
            a for a in run_dir.iterdir() if a.is_dir() and not a.is_symlink()
        ]
        configs = [a / "resolved_config.json" for a in attempts]
        configs = [c for c in configs if c.is_file()]
        if not configs:
            continue
        config = ExperimentConfig.from_dict(json.loads(configs[0].read_text()))
        probe = config.training.probe
        if probe.layers is None:
            continue
        old_segment = f"_L{probe.layer}_"
        new_segment = f"_{depth_label(probe)}_"
        if new_segment in run_dir.name:
            # Already carries its span; running this twice changes nothing.
            continue
        if old_segment not in run_dir.name:
            print(f"  ! {run_dir.name} has no {old_segment!r} segment, skipped")
            continue
        plan.append(
            {
                "old": run_dir,
                "new": OUTPUTS / run_dir.name.replace(old_segment, new_segment, 1),
                "old_segment": old_segment,
                "new_segment": new_segment,
                "layers": depth_label(probe)[1:],
            }
        )
    return plan


def rewrite_metadata(attempt: Path, item: dict, old_name: str, new_name: str) -> None:
    """Bring the run's own record of itself in line with its new name."""
    layers = item["layers"]
    info_path = attempt / "run_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        # The group carries the same depth segment as the name, and only that
        # segment changes in either.
        for key in ("run_name", "group"):
            if isinstance(info.get(key), str):
                info[key] = info[key].replace(
                    item["old_segment"], item["new_segment"], 1
                )
        info["tags"] = [
            f"layers:{layers}" if t.startswith("layer:") else t
            for t in info.get("tags", [])
        ]
        axes = info.get("axes") or {}
        if "layer" in axes:
            axes["layer"] = None
            axes["layers"] = layers
            info["axes"] = axes
        info_path.write_text(json.dumps(info, indent=2))

    for state in attempt.glob("checkpoint-*/trainer_state.json"):
        text = state.read_text()
        if old_name in text:
            state.write_text(text.replace(old_name, new_name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually rename.")
    args = parser.parse_args()

    plan = planned_renames()
    if not plan:
        print("Nothing to rename.")
        return

    clashes = [item for item in plan if item["new"].exists()]
    if clashes:
        for item in clashes:
            print(f"  REFUSING: {item['new'].name} already exists")
        raise SystemExit("target directories already exist; nothing was renamed")

    for item in plan:
        print(f"  {item['old'].name}\n->  {item['new'].name}")
    print(f"\n{len(plan)} directories")

    if not args.apply:
        print("\nDry run. Pass --apply to rename.")
        return

    for item in plan:
        old_name, new_name = item["old"].name, item["new"].name
        for attempt in item["old"].iterdir():
            if attempt.is_dir() and not attempt.is_symlink():
                rewrite_metadata(attempt, item, old_name, new_name)
        item["old"].rename(item["new"])
    print(f"\nRenamed {len(plan)} directories.")


if __name__ == "__main__":
    main()
