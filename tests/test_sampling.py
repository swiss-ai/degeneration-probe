"""The selection ladder: each rung must change exactly one thing."""

from types import SimpleNamespace

import numpy as np
import pytest

from degeneration_probe.data.sampling import Window, build_windows, compose_batches

W = 16
LENGTH, ONSET = 100, 60


def _record(prompt_id="p", *, tokens=LENGTH, positive=False, onset=ONSET, domain="d"):
    return SimpleNamespace(
        prompt_id=prompt_id,
        rollout_idx=0,
        domain=domain,
        targets=np.zeros(tokens, dtype=np.float32),
        is_positive=positive,
        onset_position=onset if positive else None,
    )


def _rng(seed=0):
    return np.random.default_rng(seed)


def _windows(records, strategy, **kwargs):
    return build_windows(records, strategy=strategy, window_size=W, rng=_rng(), **kwargs)


def test_the_reference_rung_keeps_every_token_exactly_once():
    windows = _windows([_record(tokens=LENGTH)], "all_tokens")
    covered = np.concatenate([w.positions for w in windows])
    assert sorted(covered) == list(range(LENGTH))
    assert len(windows) == int(np.ceil(LENGTH / W))


def test_every_later_rung_spends_the_same_budget_per_rollout():
    records = [_record(), _record(positive=True)]
    for strategy in ("rollout_balanced", "random_window", "frontier_window"):
        windows = _windows(records, strategy)
        assert len(windows) == len(records)
        assert all(len(window) == W for window in windows)


def test_a_rollout_shorter_than_the_budget_contributes_everything_it_has():
    short = _record(tokens=W // 2)
    for strategy in ("rollout_balanced", "random_window", "frontier_window"):
        window = _windows([short], strategy)[0]
        assert sorted(window.positions) == list(range(W // 2))


def test_the_budget_becomes_contiguous_only_from_the_third_rung():
    record = [_record()]
    scattered = _windows(record, "rollout_balanced")[0].positions
    contiguous = _windows(record, "random_window")[0].positions
    assert not np.all(np.diff(scattered) == 1)
    assert np.all(np.diff(contiguous) == 1)


def test_anchoring_puts_the_window_at_the_frontier_and_nowhere_else():
    positive = [_record(positive=True)]
    trailing = _windows(positive, "frontier_window", anchor="trailing")[0].positions
    centered = _windows(positive, "frontier_window", anchor="centered")[0].positions

    # Trailing ends at the frontier: only the run-up, nothing after "now".
    assert trailing[-1] == ONSET - 1
    assert trailing[0] == ONSET - W
    # Centered straddles it, so training sees confirmed in-pattern tokens too.
    assert centered[0] < ONSET <= centered[-1] + 1
    assert len(centered) == W


def test_a_frontier_near_the_start_still_yields_a_full_window():
    early = [_record(positive=True, onset=3)]
    window = _windows(early, "frontier_window")[0].positions
    assert window[0] == 0
    assert len(window) == W


def test_negatives_stay_randomly_placed_while_positives_are_anchored():
    records = [_record(positive=True), _record(prompt_id="n")]
    placements = {
        tuple(build_windows(records, strategy="frontier_window", window_size=W, rng=_rng(seed))[1].positions[:1])
        for seed in range(12)
    }
    assert len(placements) > 1
    anchored = {
        tuple(build_windows(records, strategy="frontier_window", window_size=W, rng=_rng(seed))[0].positions[:1])
        for seed in range(12)
    }
    assert len(anchored) == 1


def test_hard_negatives_land_where_a_rollout_looks_repetitive():
    negative = [_record()]
    hardness = {0: np.zeros(LENGTH)}
    hardness[0][70:86] = 1.0  # one clearly repetitive span

    starts = []
    for seed in range(20):
        window = build_windows(
            negative,
            strategy="frontier_window_hard_negative",
            window_size=W,
            rng=_rng(seed),
            hard_negative_fraction=1.0,
            hardness=hardness,
        )[0]
        starts.append(int(window.positions[0]))
    # Placement is proportional to how much of the repetitive span a window
    # covers, so it is a spread rather than a single point. What is guaranteed
    # is that a window covering none of it is never drawn, and that the mass
    # sits over the span.
    assert all(70 - W < start < 86 for start in starts), starts
    assert 62 <= float(np.median(starts)) <= 78, starts


def test_a_mix_ratio_leaves_some_negatives_placed_uniformly():
    negative = [_record()]
    hardness = {0: np.zeros(LENGTH)}
    hardness[0][70:86] = 1.0
    starts = {
        int(
            build_windows(
                negative,
                strategy="frontier_window_hard_negative",
                window_size=W,
                rng=_rng(seed),
                hard_negative_fraction=0.5,
                hardness=hardness,
            )[0].positions[0]
        )
        for seed in range(30)
    }
    assert any(start < 60 for start in starts)


def test_hard_negative_mining_without_a_signal_says_so():
    with pytest.raises(ValueError, match="per-token repetition signal"):
        build_windows(
            [_record()],
            strategy="frontier_window_hard_negative",
            window_size=W,
            rng=_rng(),
            hard_negative_fraction=1.0,
        )


# --- batch composition ---------------------------------------------------------


def _population(positives=10, negatives=90):
    records = [_record(f"p{i}", positive=True) for i in range(positives)]
    records += [
        _record(f"n{i}", domain="a" if i % 4 else "b") for i in range(negatives)
    ]
    return records


def test_every_batch_carries_positive_gradient():
    records = _population()
    windows = _windows(records, "frontier_window")
    batches = compose_batches(
        windows, records, batch_size=8, positive_fraction=0.25, rng=_rng()
    )
    for batch in batches:
        positives = sum(records[windows[i].record_index].is_positive for i in batch)
        assert positives >= 1
        assert len(batch) == 8


def test_negatives_are_drawn_in_proportion_to_each_domain():
    records = _population(positives=20, negatives=400)
    windows = _windows(records, "frontier_window")
    batches = compose_batches(
        windows, records, batch_size=16, positive_fraction=0.25, rng=_rng()
    )
    drawn = [
        records[windows[i].record_index].domain
        for batch in batches
        for i in batch
        if not records[windows[i].record_index].is_positive
    ]
    share = drawn.count("a") / len(drawn)
    # Three quarters of the negative rollouts are domain "a".
    assert 0.68 < share < 0.82


def test_a_prompt_cannot_flood_training_with_near_duplicates():
    records = [_record("flood", positive=True) for _ in range(20)]
    records += [_record(f"n{i}") for i in range(20)]
    windows = _windows(records, "frontier_window")
    batches = compose_batches(
        windows, records, batch_size=4, positive_fraction=0.5,
        rng=_rng(), max_rollouts_per_prompt=3,
    )
    used = sum(
        records[windows[i].record_index].prompt_id == "flood" for batch in batches for i in batch
    )
    assert used <= 3


def test_resampling_moves_the_windows_but_not_the_budget():
    records = _population()
    first = build_windows(records, strategy="random_window", window_size=W, rng=_rng(1))
    second = build_windows(records, strategy="random_window", window_size=W, rng=_rng(2))
    assert [len(w) for w in first] == [len(w) for w in second]
    assert any(
        not np.array_equal(a.positions, b.positions) for a, b in zip(first, second)
    )
