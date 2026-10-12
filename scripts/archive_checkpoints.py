"""Pack the checkpoints a run does not report into one archive, without loss.

A run saves a checkpoint every hundred steps, and a frozen run saves one head
per depth inside each of them, so a finished run holds thousands of small
files of which a handful are ever read again. Those are the checkpoints the
selection rule named, one per depth, and they stay where they are so that the
scores filed under them keep pointing at real weights. Everything else under
``checkpoint-*`` goes into a single uncompressed tar beside them, where it
costs one file instead of thousands and can be put back with ``tar -xf``.

Nothing is deleted until the archive has been read back and compared with the
originals, member by member, by ``tar --compare``. A manifest of what the
archive holds is written next to it, and a run that already has an archive is
left alone, so the script can be rerun safely.

    python scripts/archive_checkpoints.py --run-dir outputs/<run>/<attempt> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE = "checkpoints_archive.tar"
MANIFEST = "checkpoints_archive.json"
SELECTION = "checkpoint_selection.json"


def selected_pairs(run_dir: Path) -> set[tuple[int, int]]:
    payload = json.loads((run_dir / SELECTION).read_text(encoding="utf-8"))
    return {(int(d["layer"]), int(d["selected_step"])) for d in payload["depths"]}


def kept_paths(run_dir: Path, pairs: set[tuple[int, int]], multi_depth: bool) -> set[str]:
    """What stays on disk, relative to the run directory."""
    if multi_depth:
        return {f"checkpoint-{step}/layer_{layer:02d}" for layer, step in pairs}
    return {f"checkpoint-{step}" for _, step in pairs}


def step_checkpoints(run_dir: Path) -> list[Path]:
    """The trainer's own checkpoints, ``checkpoint-<step>``; the composite
    ``checkpoint-selected`` directory of links is not one of them."""
    return sorted(
        p for p in run_dir.glob("checkpoint-*")
        if p.is_dir() and not p.is_symlink() and p.name[len("checkpoint-"):].isdigit()
    )


def archivable(run_dir: Path, keep: set[str], multi_depth: bool) -> list[str]:
    """Every checkpoint entry that is not kept, as a path relative to the run.

    A checkpoint that holds a kept depth stays a complete checkpoint, trainer
    state included, so a run could still resume from it; only its unselected
    depths are archived. A checkpoint with no kept depth is archived whole.
    """
    members: list[str] = []
    for checkpoint in step_checkpoints(run_dir):
        rel = checkpoint.name
        if not multi_depth:
            if rel not in keep:
                members.append(rel)
            continue
        kept_here = any(k.startswith(rel + "/") for k in keep)
        if not kept_here:
            members.append(rel)
            continue
        for entry in sorted(checkpoint.iterdir()):
            entry_rel = f"{rel}/{entry.name}"
            if entry.is_dir() and entry_rel not in keep:
                members.append(entry_rel)
    return members


def count_files(run_dir: Path, members: list[str]) -> int:
    total = 0
    for member in members:
        path = run_dir / member
        total += sum(1 for p in path.rglob("*") if p.is_file()) if path.is_dir() else 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    if (run_dir / ARCHIVE).exists():
        print(f"skip   {run_dir.name}: {ARCHIVE} already exists")
        return
    if not (run_dir / SELECTION).is_file():
        raise SystemExit(f"{run_dir} has no {SELECTION}; nothing says which checkpoints to keep")
    pairs = selected_pairs(run_dir)
    sample = next(run_dir.glob("checkpoint-*/layer_*"), None)
    multi_depth = sample is not None
    keep = kept_paths(run_dir, pairs, multi_depth)
    for rel in keep:
        if not (run_dir / rel).exists():
            raise SystemExit(f"selected checkpoint {rel} is missing under {run_dir}")
    members = archivable(run_dir, keep, multi_depth)
    if not members:
        print(f"skip   {run_dir.name}: nothing to archive")
        return
    files = count_files(run_dir, members)
    print(
        f"{'would ' if args.dry_run else ''}archive {run_dir.name}: {len(members)} entries, "
        f"{files} files; keeping {len(keep)} selected checkpoint dirs"
    )
    if args.dry_run:
        return

    archive = run_dir / ARCHIVE
    staged = run_dir / (ARCHIVE + ".partial")
    listing = run_dir / (ARCHIVE + ".members")
    listing.write_text("\n".join(members) + "\n", encoding="utf-8")
    subprocess.run(
        ["tar", "-cf", str(staged), "-C", str(run_dir), "-T", str(listing)], check=True
    )
    # Read the archive back and compare every member with the files on disk;
    # a non-zero exit means something differs, and nothing is removed.
    subprocess.run(["tar", "--compare", "-f", str(staged), "-C", str(run_dir)], check=True)
    archived = subprocess.run(
        ["tar", "-tf", str(staged)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    archived_files = sum(1 for m in archived if not m.endswith("/"))
    if archived_files != files:
        raise SystemExit(f"archive holds {archived_files} files but {files} were listed; not removing")
    staged.rename(archive)
    listing.unlink()

    for member in members:
        path = run_dir / member
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    for checkpoint in step_checkpoints(run_dir):
        if not any(checkpoint.iterdir()):
            checkpoint.rmdir()

    manifest = {
        "archive": ARCHIVE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "multi_depth": multi_depth,
        "kept": sorted(keep),
        "members": members,
        "files": files,
        "restore": f"tar -xf {ARCHIVE} -C {run_dir}",
    }
    (run_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done   {run_dir.name}: {files} files -> {ARCHIVE} ({archive.stat().st_size / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
