import numpy as np
import pandas as pd
import pytest

from degeneration_probe.evaluation.scores import (
    build_scores,
    read_scores,
    validate_scores,
    write_scores,
)


def _record(name="p0", scores=(0.1, 0.9, 0.4), onset=None, split="val"):
    scores = np.asarray(scores, dtype=np.float32)
    return {
        "prompt_id": name,
        "rollout_idx": 0,
        "domain": "d",
        "split": split,
        "stop_reason": "length" if onset is not None else "eos",
        "num_tokens": int(scores.size),
        "onset_position": float(onset) if onset is not None else None,
        "is_positive": onset is not None,
        "scores": scores,
    }


def test_a_score_table_round_trips_through_disk(tmp_path):
    frame = build_scores([_record(), _record("p1", onset=1)])
    path = write_scores(frame, tmp_path / "val.parquet")
    reloaded = read_scores(path)
    assert len(reloaded) == 2
    assert np.allclose(reloaded["scores"].iloc[0], [0.1, 0.9, 0.4], atol=1e-3)
    assert reloaded["is_positive"].tolist() == [False, True]


def test_reading_can_restrict_to_one_split(tmp_path):
    frame = build_scores(
        [_record("a", split="val"), _record("b", split="test_indomain")]
    )
    path = write_scores(frame, tmp_path / "all.parquet")
    assert read_scores(path, split="val")["prompt_id"].tolist() == ["a"]
    with pytest.raises(ValueError, match="No scored rollouts"):
        read_scores(path, split="absent")


def test_a_score_per_token_is_required():
    bad = _record()
    bad["num_tokens"] = 5
    with pytest.raises(ValueError, match="3 scores for a 5-token rollout"):
        build_scores([bad])


def test_scores_outside_zero_to_one_are_refused():
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        build_scores([_record(scores=(0.5, 1.4))])
    with pytest.raises(ValueError, match="non-finite"):
        build_scores([_record(scores=(0.5, np.nan))])


def test_a_positive_rollout_needs_a_frontier_inside_it():
    with pytest.raises(ValueError, match="needs an onset position"):
        frame = pd.DataFrame([{**_record(), "is_positive": True}])
        validate_scores(frame[list(_record().keys())])
    with pytest.raises(ValueError, match="outside a 3-token rollout"):
        build_scores([_record(onset=7)])


def test_a_rollout_may_not_appear_twice():
    with pytest.raises(ValueError, match="Duplicate rollout keys"):
        build_scores([_record("p0"), _record("p0")])


def test_an_interrupted_write_leaves_no_partial_table(tmp_path, monkeypatch):
    frame = build_scores([_record()])
    path = tmp_path / "val.parquet"

    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)
    with pytest.raises(RuntimeError):
        write_scores(frame, path)
    assert not path.exists()
    assert not list(tmp_path.glob(".tmp_scores_*"))
