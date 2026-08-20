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


def _score_row(prompt_id, rollout_idx, *, positive, stop_reason, onset, tokens=8):
    return {
        "prompt_id": prompt_id,
        "rollout_idx": rollout_idx,
        "domain": "d",
        "split": "val",
        "stop_reason": stop_reason,
        "num_tokens": tokens,
        "onset_position": onset,
        "is_positive": positive,
        "scores": np.linspace(0.0, 1.0, tokens, dtype=np.float32),
    }


def test_unjudged_rollouts_selects_only_the_unruled(tmp_path):
    from degeneration_probe.evaluation.scores import unjudged_rollouts

    labels = pd.DataFrame(
        {
            "prompt_id": ["a", "b", "c", "d", "e"],
            "rollout_idx": [0, 1, 2, 3, 4],
            "onset_resolution": ["ok", "judge_failed", "not_found", "not_degenerating", None],
        }
    )
    path = tmp_path / "onset_labels.parquet"
    labels.to_parquet(path, index=False)
    # not_degenerating was ruled on and stays a negative; ok is a positive; a
    # null resolution is an answer that ended at an end-of-sequence token.
    assert unjudged_rollouts(path) == {("b", 1), ("c", 2)}


def test_drop_unjudged_reports_how_many_went():
    from degeneration_probe.evaluation.scores import drop_unjudged

    frame = pd.DataFrame(
        [
            _score_row("a", 0, positive=True, stop_reason="length", onset=2.0),
            _score_row("b", 1, positive=False, stop_reason="length", onset=None),
            _score_row("c", 2, positive=False, stop_reason="eos", onset=None),
        ]
    )
    kept, dropped = drop_unjudged(frame, {("b", 1)})
    assert dropped == 1
    assert list(kept["prompt_id"]) == ["a", "c"]
    # An empty exclusion set must not copy or reindex anything.
    same, none_dropped = drop_unjudged(frame, set())
    assert none_dropped == 0
    assert len(same) == 3


def test_read_scores_excludes_unjudged_rollouts(tmp_path):
    from degeneration_probe.evaluation.scores import build_scores, read_scores, write_scores

    frame = build_scores(
        [
            _score_row("a", 0, positive=True, stop_reason="length", onset=2.0),
            _score_row("b", 1, positive=False, stop_reason="length", onset=None),
            _score_row("c", 2, positive=False, stop_reason="eos", onset=None),
        ]
    )
    path = write_scores(frame, tmp_path / "val.parquet")
    assert len(read_scores(path, split="val")) == 3
    trimmed = read_scores(path, split="val", unjudged={("b", 1)})
    assert list(trimmed["prompt_id"]) == ["a", "c"]


def test_read_scores_refuses_a_wholly_unjudged_table(tmp_path):
    from degeneration_probe.evaluation.scores import build_scores, read_scores, write_scores

    frame = build_scores(
        [_score_row("b", 1, positive=False, stop_reason="length", onset=None)]
    )
    path = write_scores(frame, tmp_path / "val.parquet")
    with pytest.raises(ValueError, match="unjudged"):
        read_scores(path, split="val", unjudged={("b", 1)})
