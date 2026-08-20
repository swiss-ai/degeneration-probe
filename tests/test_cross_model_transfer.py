"""What may and may not be carried onto another model's dataset.

Transfer asks whether a head fitted to one model still separates degenerating
rollouts on a model it never saw. Two things decide whether a given checkpoint
can answer that at all, and both are easy to get silently wrong: a LoRA run
carries weights that only mean something in the layers they were fitted to, and
an adapted run has to read the *target* model, since re-reading the source would
transfer nothing while still producing a full, plausible score file.
"""

from pathlib import Path

import pytest
import yaml

from scripts.evaluate_cross_model_transfer import load_dataset_config, target_model_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG_DIR = REPO_ROOT / "configs" / "dataset"
DATASET_CONFIGS = sorted(DATASET_CONFIG_DIR.glob("degeneration-dataset-*.yaml"))


@pytest.mark.parametrize("config_path", DATASET_CONFIGS, ids=lambda p: p.stem)
def test_the_target_model_is_the_one_the_dataset_was_generated_with(config_path):
    dataset_config = load_dataset_config(config_path)
    build = yaml.safe_load((REPO_ROOT / dataset_config.build_config).read_text())

    resolved = target_model_config(dataset_config, dtype="bfloat16")

    assert resolved.name == build["model_name"]
    assert resolved.tokenizer_name == (build.get("tokenizer_name") or None)


def test_the_run_dtype_is_carried_rather_than_the_build_s():
    """Transfer changes whose representation is read, and nothing else."""
    dataset_config = load_dataset_config(DATASET_CONFIGS[0])
    assert target_model_config(dataset_config, dtype="float16").dtype == "float16"
    assert target_model_config(dataset_config, dtype="bfloat16").dtype == "bfloat16"


def test_every_dataset_config_names_a_build_that_exists():
    """A missing build config would surface as a confusing model-load failure."""
    for config_path in DATASET_CONFIGS:
        dataset_config = load_dataset_config(config_path)
        assert (REPO_ROOT / dataset_config.build_config).is_file(), dataset_config.build_config
