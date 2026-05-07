import json
import importlib.util
import sys
import types
from pathlib import Path


def _load_repetition_converter():
    repo_root = Path(__file__).resolve().parents[1]

    datasets_stub = types.ModuleType("datasets")
    datasets_stub.Dataset = list
    sys.modules.setdefault("datasets", datasets_stub)

    feature_probes_pkg = types.ModuleType("feature_probes")
    feature_probes_pkg.__path__ = [str(repo_root / "feature_probes")]
    sys.modules.setdefault("feature_probes", feature_probes_pkg)

    data_pkg = types.ModuleType("feature_probes.data")
    data_pkg.__path__ = [str(repo_root / "feature_probes" / "data")]
    sys.modules.setdefault("feature_probes.data", data_pkg)

    types_spec = importlib.util.spec_from_file_location(
        "feature_probes.types",
        repo_root / "feature_probes" / "types.py",
    )
    types_module = importlib.util.module_from_spec(types_spec)
    sys.modules["feature_probes.types"] = types_module
    types_spec.loader.exec_module(types_module)

    converters_spec = importlib.util.spec_from_file_location(
        "feature_probes.data.converters",
        repo_root / "feature_probes" / "data" / "converters.py",
    )
    converters_module = importlib.util.module_from_spec(converters_spec)
    sys.modules["feature_probes.data.converters"] = converters_module
    converters_spec.loader.exec_module(converters_module)

    return converters_module.prepare_repetition_instruct_token_level_dataset


prepare_repetition_instruct_token_level_dataset = _load_repetition_converter()


def test_repetition_converter_uses_chunk_summary_repetition_scores():
    rows = [
        {
            "prompt": "Write a short answer.",
            "generated_text": "A short short answer.",
            "degenerating": True,
            "chunk_summary": [
                {"token_index": 2, "repetition": 0.3, "degenerating": False},
                {"token_index": 0, "repetition": 0.1, "degenerating": False},
                {"token_index": 1, "repetition": 0.2, "degenerating": True},
            ],
        }
    ]

    item = prepare_repetition_instruct_token_level_dataset(rows)[0]

    assert item.prompt == "Write a short answer."
    assert item.completion == "A short short answer."
    assert item.token_labels == [0.1, 0.2, 0.3]


def test_repetition_converter_accepts_json_chunk_summary_and_fills_gaps():
    rows = [
        {
            "prompt": "Prompt",
            "generated_text": "Completion",
            "chunk_summary": json.dumps(
                [
                    {"token_index": 0, "repetition": 0.4, "degenerating": False},
                    {"token_index": 2, "repetition": 0.6, "degenerating": True},
                ]
            ),
        }
    ]

    item = prepare_repetition_instruct_token_level_dataset(rows)[0]

    assert item.token_labels == [0.4, -100.0, 0.6]


def test_repetition_converter_treats_null_repetition_as_ignored_label():
    rows = [
        {
            "prompt": "Prompt",
            "generated_text": "Completion",
            "chunk_summary": [
                {"token_index": 0, "repetition": 0.4, "degenerating": False},
                {"token_index": 1, "repetition": None, "degenerating": False},
                {"token_index": 2, "repetition": 0.6, "degenerating": True},
            ],
        }
    ]

    item = prepare_repetition_instruct_token_level_dataset(rows)[0]

    assert item.token_labels == [0.4, -100.0, 0.6]
