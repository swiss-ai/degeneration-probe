"""Make the vendored fork at third_party/hallucination_probes importable.

Call `ensure_fork_on_syspath()` before importing from `probe.*`, `utils.*`,
or `degeneration.*` (the fork's top-level packages).
"""

from __future__ import annotations

import sys
from pathlib import Path

FORK_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "third_party" / "hallucination_probes"
)


def ensure_fork_on_syspath() -> None:
    p = str(FORK_ROOT)
    if not FORK_ROOT.exists():
        raise SystemExit(
            f"Fork not found at {FORK_ROOT}. "
            f"Run `git submodule update --init` to populate it."
        )
    if p not in sys.path:
        sys.path.insert(0, p)
