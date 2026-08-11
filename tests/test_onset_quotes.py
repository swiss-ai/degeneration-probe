"""Locating the judge's onset quote in a rollout's tokens."""

import pandas as pd
import pytest

from degeneration_probe.dataset_gen.onset_quotes import (
    _classify,
    _quote_of,
    load_judge_results,
    locate_quote,
)
from degeneration_probe.dataset_gen.onset_labels import resolve_onset_position


class LetterTokenizer:
    """One token per letter, so a token index is a character index."""

    def decode(self, tokens):
        return "".join(chr(int(t)) for t in tokens)


def _ids(text):
    return [ord(character) for character in text]


def test_a_quote_resolves_to_the_token_it_starts_at():
    span = locate_quote("cde", _ids("abcdefg"), LetterTokenizer())
    assert (span.start, span.stop) == (2, 5)


def test_a_repeated_quote_resolves_to_its_first_occurrence():
    """The onset is where the loop began, not where it was still going.

    A quote naming the repeated unit occurs many times by construction, so the
    earliest occurrence is the only one that can be the onset.
    """
    span = locate_quote("abc", _ids("xxabcabcabc"), LetterTokenizer())
    assert span.start == 2


def test_a_quote_that_is_not_there_resolves_to_nothing():
    # Never a fallback position: a quote the tokens do not contain is a failure
    # to be recorded, not an onset to be guessed at.
    assert locate_quote("zzz", _ids("abcdefg"), LetterTokenizer()) is None
    assert locate_quote("", _ids("abc"), LetterTokenizer()) is None


def test_a_missing_quote_is_recognised_despite_being_truthy():
    """A quote absent from a dataframe arrives as NaN, and NaN is truthy."""
    assert _quote_of({"onset_quote": float("nan")}) == ""
    assert _quote_of({"onset_quote": None}) == ""
    assert _quote_of({}) == ""
    assert _quote_of({"onset_quote": "loop"}) == "loop"


@pytest.mark.parametrize(
    "row,expected",
    [
        ({"status": None, "is_degenerating": None, "onset_quote": None}, "unjudged"),
        ({"status": "failed", "is_degenerating": None, "onset_quote": None}, "judge_failed"),
        ({"status": "ok", "is_degenerating": False, "onset_quote": None}, "not_degenerating"),
        ({"status": "ok", "is_degenerating": True, "onset_quote": float("nan")}, "empty_quote"),
        ({"status": "ok", "is_degenerating": True, "onset_quote": "loop"}, "ok"),
    ],
)
def test_every_rollout_without_a_position_says_why(row, expected):
    assert _classify(row) == expected


def _judge_dir(tmp_path, frames):
    directory = tmp_path / "llm_judge"
    directory.mkdir(parents=True)
    for backend, frame in frames.items():
        frame.to_parquet(directory / f"results_{backend}.parquet")

    class Config:
        pass

    config = Config()
    import degeneration_probe.dataset_gen.paths as paths

    original = paths.llm_judge_dir
    paths.llm_judge_dir = lambda _config: directory
    return config, lambda: setattr(paths, "llm_judge_dir", original)


def test_a_refusal_never_displaces_another_backend_s_verdict(tmp_path):
    """Pooling backends must not lose information.

    One backend refusing a rollout says nothing about it. If that refusal wins
    the merge, a known verdict is reported as an unexplained failure.
    """
    refused = pd.DataFrame(
        {
            "prompt_id": ["p", "q"],
            "rollout_idx": [0, 0],
            "status": ["failed", "failed"],
            "is_degenerating": [None, None],
            "onset_quote": [None, None],
        }
    )
    answered = pd.DataFrame(
        {
            "prompt_id": ["p", "q"],
            "rollout_idx": [0, 0],
            "status": ["ok", "ok"],
            "is_degenerating": [True, False],
            "onset_quote": ["the loop", None],
        }
    )
    # Alphabetically first, so it wins any merge that only preserves order.
    config, restore = _judge_dir(tmp_path, {"aaa_refused": refused, "zzz_answered": answered})
    try:
        pooled = load_judge_results(config).set_index("prompt_id")
    finally:
        restore()
    assert pooled.loc["p", "status"] == "ok"
    assert pooled.loc["p", "onset_quote"] == "the loop"
    # An "ok, not degenerating" verdict also beats a refusal.
    assert pooled.loc["q", "status"] == "ok"


def test_the_label_seam_reads_the_cached_position():
    assert resolve_onset_position({"onset_quote_position": 314}) == 314


def test_the_label_seam_invents_nothing_when_a_quote_did_not_resolve():
    """A cached table that resolved to nothing is an answer, not a gap.

    Falling through to a live search here would re-run the work that already
    failed, and quietly disagree with the reason recorded beside it.
    """
    assert resolve_onset_position({"onset_quote_position": None, "onset_quote": "loop"}) is None
    assert (
        resolve_onset_position({"onset_quote_position": float("nan"), "onset_quote": "loop"})
        is None
    )


def test_an_unknown_onset_metric_is_refused():
    """A misspelled metric must fail rather than fall through to the default,
    which would attach the wrong provenance to a whole corpus of labels."""
    with pytest.raises(ValueError, match="Unknown onset_metric"):
        resolve_onset_position({"onset_quote_position": 1}, metric="onset_qoute")
