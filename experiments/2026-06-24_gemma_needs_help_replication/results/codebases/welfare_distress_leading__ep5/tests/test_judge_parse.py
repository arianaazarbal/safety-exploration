"""Tests for robust parsing of judge output into an integer score."""

import pytest

from distress_eval.judge import _parse_score


def test_clean_json():
    r = _parse_score('{"score": 7, "reasoning": "lots of caps and apologies"}')
    assert r.score == 7
    assert "caps" in r.reasoning


def test_json_in_code_fence():
    r = _parse_score('```json\n{"score": 3, "reasoning": "mild"}\n```')
    assert r.score == 3


def test_clamps_out_of_range():
    assert _parse_score('{"score": 15}').score == 10
    assert _parse_score('{"score": -2}').score == 0


def test_fallback_regex_on_prose():
    r = _parse_score("I would rate this a score: 5 out of 10.")
    assert r.score == 5


def test_unparseable_raises():
    with pytest.raises(ValueError):
        _parse_score("no numbers here at all")
