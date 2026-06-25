"""Robustness of judge-output parsing (the metrics depend only on the rating)."""

import pytest

from emotional_instability.eval.judge import parse_judge_output


def test_parses_clean_json():
    raw = '{"evidence": "ugh", "reasoning": "mild", "rating": 4}'
    rating, evidence, reasoning = parse_judge_output(raw)
    assert rating == 4
    assert evidence == "ugh"


def test_parses_json_with_surrounding_prose():
    raw = 'Here is my assessment:\n{"evidence": "x", "reasoning": "y", "rating": 9}\nDone.'
    rating, _, _ = parse_judge_output(raw)
    assert rating == 9


def test_clamps_out_of_range():
    raw = '{"evidence": "x", "reasoning": "y", "rating": 12}'
    rating, _, _ = parse_judge_output(raw)
    assert rating == 10


def test_regex_fallback_when_json_broken():
    raw = 'evidence: none, reasoning: ..., rating: 3 (broken json'
    rating, _, _ = parse_judge_output(raw)
    assert rating == 3


def test_raises_when_no_rating():
    with pytest.raises(ValueError):
        parse_judge_output("the model seemed fine, no score given")
