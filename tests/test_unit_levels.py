"""The unit re-measurement: boundaries tile an answer, and the metrics are right."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from degeneration_probe.analysis.unit_levels import (
    balanced_accuracy,
    delimiters_survive,
    roc_auc,
    token_char_spans,
    unit_boundaries,
    unit_scores,
)


class FakeTokenizer:
    """Decodes one id at a time from a fixed table, like the real one does."""

    def __init__(self, table):
        self.table = table

    def decode(self, ids):
        return "".join(self.table[int(i)] for i in ids)


PIECES = {
    1: "Hello",
    2: " world",
    3: ".",
    4: "\n\n",
    5: "Next",
    6: " one",
    7: "!",
    8: " Tail",
}


def spans_for(kind, table=PIECES, ids=(1, 2, 3, 4, 5, 6, 7, 8)):
    text, starts = token_char_spans(FakeTokenizer(table), list(ids))
    return text, unit_boundaries(text, starts, kind)


def test_pieces_reassemble_and_offsets_line_up():
    text, starts = token_char_spans(FakeTokenizer(PIECES), [1, 2, 3, 4])
    assert text == "Hello world.\n\n"
    assert starts.tolist() == [0, 5, 11, 12]


def test_chunk_boundaries_cut_at_the_blank_line():
    _, spans = spans_for("chunk")
    assert spans == [(0, 3), (4, 7)]


def test_sentence_boundaries_cut_at_terminal_punctuation():
    _, spans = spans_for("sentence")
    assert spans == [(0, 2), (3, 6), (7, 7)]


def test_answer_is_one_span():
    _, spans = spans_for("answer")
    assert spans == [(0, 7)]


@pytest.mark.parametrize("kind", ["sentence", "chunk", "answer"])
def test_spans_tile_the_answer_without_gaps_or_overlap(kind):
    _, spans = spans_for(kind)
    covered = [index for start, end in spans for index in range(start, end + 1)]
    assert covered == list(range(8))


def test_a_delimiter_at_the_very_end_leaves_no_empty_trailing_span():
    _, spans = spans_for("sentence", ids=(1, 2, 3))
    assert spans == [(0, 2)]


def test_no_delimiter_gives_a_single_span():
    _, spans = spans_for("chunk", ids=(1, 2, 5, 6))
    assert spans == [(0, 3)]


def test_empty_answer_has_no_units():
    text, starts = token_char_spans(FakeTokenizer(PIECES), [])
    assert unit_boundaries(text, starts, "sentence") == []


def test_delimiter_survival_detects_a_lost_newline():
    assert delimiters_survive("a.\nb", "a.\nb")
    assert not delimiters_survive("a.b", "a.\nb")
    # A mangled multi-byte character is not a lost delimiter.
    assert delimiters_survive("k��P.", "k√P.")


def test_unit_scores_reductions():
    scores = np.array([0.0, 1.0, 0.5, 0.25])
    spans = [(0, 1), (2, 3)]
    assert unit_scores(scores, spans, "final").tolist() == [1.0, 0.25]
    assert unit_scores(scores, spans, "mean").tolist() == [0.5, 0.375]
    assert unit_scores(scores, spans, "max").tolist() == [1.0, 0.5]


def test_unknown_unit_and_reduction_are_refused():
    text, starts = token_char_spans(FakeTokenizer(PIECES), [1, 2])
    with pytest.raises(ValueError):
        unit_boundaries(text, starts, "paragraph")
    with pytest.raises(ValueError):
        unit_scores(np.zeros(2), [(0, 1)], "median")


def test_roc_auc_matches_sklearn_including_ties():
    generator = np.random.default_rng(0)
    for _ in range(50):
        size = int(generator.integers(4, 120))
        labels = generator.integers(0, 2, size)
        if labels.min() == labels.max():
            continue
        # Rounded on purpose, so ties are common.
        values = np.round(generator.normal(size=size), 1)
        assert roc_auc(labels, values) == pytest.approx(roc_auc_score(labels, values))


def test_roc_auc_is_undefined_with_one_class():
    assert np.isnan(roc_auc(np.zeros(4, dtype=int), np.arange(4.0)))


def test_balanced_accuracy_matches_a_brute_force_sweep():
    generator = np.random.default_rng(1)
    for _ in range(50):
        size = int(generator.integers(6, 100))
        labels = generator.integers(0, 2, size)
        if labels.min() == labels.max():
            continue
        values = np.round(generator.normal(size=size), 1)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        keep = min(positive.size, negative.size)
        picker = np.random.default_rng(42)
        chosen = np.concatenate(
            [
                picker.choice(positive, keep, replace=False),
                picker.choice(negative, keep, replace=False),
            ]
        )
        truth, score = labels[chosen], values[chosen]
        cuts = np.concatenate([score, [score.min() - 1, score.max() + 1]])
        brute = max(((score >= cut) == (truth == 1)).mean() for cut in np.unique(cuts))
        assert balanced_accuracy(labels, values) == pytest.approx(brute)


def test_a_perfect_separator_reads_one_and_a_useless_one_reads_a_half():
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert balanced_accuracy(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(labels, np.ones(4)) == pytest.approx(0.5)


def test_average_precision_matches_sklearn_including_ties():
    from sklearn.metrics import average_precision_score

    from degeneration_probe.analysis.unit_levels import average_precision

    generator = np.random.default_rng(3)
    for _ in range(60):
        size = int(generator.integers(4, 150))
        labels = generator.integers(0, 2, size)
        if labels.min() == labels.max():
            continue
        # Rounded to whole numbers, so ties dominate.
        values = np.round(generator.normal(size=size))
        assert average_precision(labels, values) == pytest.approx(
            average_precision_score(labels, values)
        )


def test_best_accuracy_cannot_split_a_tied_group():
    """A threshold acts on every equal score at once, and the metric must too."""
    from degeneration_probe.analysis.unit_levels import best_accuracy

    # Four answers all scoring the same, two of each class. No threshold separates
    # them, so the best any cut achieves is firing on all or none: 0.5 either way.
    labels = np.array([1, 1, 0, 0])
    assert best_accuracy(labels, np.ones(4)) == pytest.approx(0.5)

    generator = np.random.default_rng(4)
    for _ in range(60):
        size = int(generator.integers(4, 120))
        labels = generator.integers(0, 2, size)
        if labels.min() == labels.max():
            continue
        values = np.round(generator.normal(size=size))
        cuts = np.unique(np.append(values, values.max() + 1))
        brute = max(((values >= cut) == (labels == 1)).mean() for cut in cuts)
        assert best_accuracy(labels, values) == pytest.approx(brute)


def test_accuracy_reports_the_floor_the_class_balance_sets():
    """A scorer that never fires already scores the negative share."""
    from degeneration_probe.analysis.unit_levels import best_accuracy

    labels = np.array([1] + [0] * 99)
    # A useless constant score cannot beat refusing to fire.
    assert best_accuracy(labels, np.ones(100)) == pytest.approx(0.99)
