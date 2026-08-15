"""Pin the set of rollouts that evaluation during training reads.

Which rollouts a run validates on has to be identical between a run and any
later replay of its checkpoints, or the two produce different numbers for
reasons that have nothing to do with the probe. Deriving that set at runtime
from a seeded sample makes it depend on record ordering, on the sampling code
and on the split file all at once, and none of those announce themselves when
they change.

So the set is written down instead, as the identity of each rollout: the three
fields that locate its cached activations. Everything else about a rollout,
including where its loop begins, is looked up through the function that owns
that definition rather than copied here, so this file can never disagree with
it.

    python scripts/write_validation_rollouts.py

Writes configs/dataset/validation_rollouts_<dataset>.csv, which is small enough
to keep in version control, so any change to the evaluation population shows up
as a diff rather than as a surprise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import ExperimentConfig, LabelConfig
from degeneration_probe.data.dataset import load_degeneration_records

IDENTITY = ["domain", "prompt_id", "rollout_idx"]


def build(config_name: str = "main") -> tuple[pd.DataFrame, ExperimentConfig]:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        composed = compose(config_name=config_name)
    experiment = ExperimentConfig.from_dict(OmegaConf.to_container(composed, resolve=True))
    # Evaluation never thins its population, whatever a run was trained on.
    experiment.dataset.sampling.evaluation_negative_rollouts_per_positive = None
    records = load_degeneration_records(
        experiment.dataset,
        split=experiment.dataset.splits.validation,
        label_config=LabelConfig(),
        training=False,
    )
    frame = pd.DataFrame(
        [
            {
                "domain": record.domain,
                "prompt_id": record.prompt_id,
                "rollout_idx": int(record.rollout_idx),
            }
            for record in records
        ]
    )
    return frame.sort_values(IDENTITY).reset_index(drop=True), experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="main")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    frame, experiment = build(args.config_name)
    out = args.out or (
        REPO_ROOT
        / "configs"
        / "dataset"
        / f"validation_rollouts_{experiment.dataset.short_name}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"{len(frame)} rollouts -> {out}")
    print(frame.groupby("domain").size().to_string())


if __name__ == "__main__":
    main()
