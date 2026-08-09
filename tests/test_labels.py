"""The three label families, and the boundary between a label and a loss."""

import numpy as np
import pytest

from degeneration_probe.config import LabelConfig
from degeneration_probe.data.labels import (
    derive_targets,
    frontier_hard_targets,
    frontier_soft_targets,
    produces_binary_targets,
    required_signal,
    token_signal_targets,
)

LENGTH, ONSET = 20, 12


def test_the_default_family_is_a_step_at_the_frontier():
    targets = frontier_hard_targets(LENGTH, stop_reason="length", onset_position=ONSET)
    assert targets[:ONSET] == [0.0] * ONSET
    assert targets[ONSET:] == [1.0] * (LENGTH - ONSET)


def test_a_horizon_brings_the_step_forward_by_exactly_that_many_tokens():
    targets = frontier_hard_targets(
        LENGTH, stop_reason="length", onset_position=ONSET, horizon=5
    )
    assert sum(targets) == LENGTH - (ONSET - 5)
    assert targets[ONSET - 5] == 1.0
    assert targets[ONSET - 6] == 0.0


def test_a_horizon_past_the_start_labels_the_whole_rollout():
    targets = frontier_hard_targets(
        LENGTH, stop_reason="length", onset_position=ONSET, horizon=ONSET + 50
    )
    assert targets == [1.0] * LENGTH


def test_a_rollout_that_ended_cleanly_is_negative_throughout_every_frontier_family():
    for family in (frontier_hard_targets, frontier_soft_targets):
        assert family(LENGTH, stop_reason="eos", onset_position=None) == [0.0] * LENGTH


def test_a_truncated_rollout_with_no_frontier_is_dropped_rather_than_invented():
    assert frontier_hard_targets(LENGTH, stop_reason="length", onset_position=None) is None
    assert frontier_soft_targets(LENGTH, stop_reason="length", onset_position=None) is None


def test_the_soft_family_is_one_inside_the_pattern_and_decays_backwards():
    targets = np.array(
        frontier_soft_targets(
            LENGTH, stop_reason="length", onset_position=ONSET, decay="exponential", decay_length=4
        )
    )
    assert np.all(targets[ONSET:] == 1.0)
    # Strictly decreasing back through the run-up, and never negative.
    run_up = targets[:ONSET]
    assert np.all(np.diff(run_up) > 0)
    assert run_up.min() > 0.0
    assert targets[ONSET - 4] == pytest.approx(np.exp(-1.0))


def test_a_linear_decay_reaches_zero_at_the_decay_length_and_stays_there():
    targets = np.array(
        frontier_soft_targets(
            LENGTH, stop_reason="length", onset_position=ONSET, decay="linear", decay_length=4
        )
    )
    assert targets[ONSET - 2] == pytest.approx(0.5)
    assert targets[ONSET - 4] == pytest.approx(0.0)
    assert np.all(targets[: ONSET - 4] == 0.0)


def test_the_soft_family_becomes_the_hard_one_as_the_decay_shortens():
    sharp = frontier_soft_targets(
        LENGTH, stop_reason="length", onset_position=ONSET, decay="linear", decay_length=1e-6
    )
    assert sharp == frontier_hard_targets(LENGTH, stop_reason="length", onset_position=ONSET)


def test_a_token_signal_keeps_undefined_positions_masked():
    targets = token_signal_targets([0.1, None, 0.3], 4)
    assert targets[0] == 0.1
    assert np.isnan(targets[1])
    assert targets[2] == 0.3
    assert np.isnan(targets[3])


def test_the_dispatcher_routes_each_family_and_reports_what_it_needs():
    hard = LabelConfig(family="frontier_hard", horizon=3)
    soft = LabelConfig(family="frontier_soft", decay="linear", decay_length=8)
    signal = LabelConfig(family="token_signal", signal="entropy")

    assert required_signal(hard) is None
    assert required_signal(signal) == "entropy"
    assert produces_binary_targets(hard) is True
    assert produces_binary_targets(soft) is False

    common = dict(num_tokens=LENGTH, stop_reason="length", onset_position=ONSET)
    assert derive_targets(hard, **common)[ONSET - 3] == 1.0
    assert 0.0 < derive_targets(soft, **common)[ONSET - 4] < 1.0
    assert derive_targets(signal, **common, signal=[0.5] * LENGTH)[0] == 0.5


def test_a_signal_family_without_its_column_says_so():
    with pytest.raises(ValueError, match="needs the 'repetition_score' column"):
        derive_targets(
            LabelConfig(family="token_signal"),
            num_tokens=LENGTH,
            stop_reason="eos",
            onset_position=None,
        )


def test_a_class_weight_is_refused_for_a_family_that_is_not_binary():
    from degeneration_probe.config import BceLossConfig, LossConfig, ProbeConfig, TrainingConfig

    with pytest.raises(ValueError, match="has no meaning for label family"):
        TrainingConfig(
            task="degeneration",
            short_name="d",
            probe=ProbeConfig(id="p"),
            label=LabelConfig(family="frontier_soft"),
            loss=LossConfig(name="bce", bce=BceLossConfig(use_pos_weight=True)),
        )


def test_bad_label_settings_are_rejected_before_a_run_starts():
    for kwargs, message in [
        ({"family": "invented"}, "family must be one of"),
        ({"horizon": -1}, "horizon must be non-negative"),
        ({"decay": "quadratic"}, "decay must be one of"),
        ({"decay_length": 0}, "decay_length must be positive"),
        ({"signal": "vibes"}, "signal must be one of"),
    ]:
        with pytest.raises(ValueError, match=message):
            LabelConfig(**kwargs)
