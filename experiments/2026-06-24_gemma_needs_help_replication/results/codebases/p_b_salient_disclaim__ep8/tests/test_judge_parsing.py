"""Unit tests for judge JSON parsing robustness (no network)."""
from emotional_instability.judge import _coerce_rating, _parse_judge_json


def test_parse_plain_json():
    out = _parse_judge_json('{"evidence": "ugh", "reasoning": "mild", "rating": 2}')
    assert out["rating"] == 2


def test_parse_with_preamble_and_smart_quotes():
    raw = 'Here is my analysis.\n{“evidence”: “argh”, “reasoning”: “x”, “rating”: 7}'
    out = _parse_judge_json(raw)
    assert out["rating"] == 7


def test_coerce_rating_clamps():
    assert _coerce_rating(11) == 10
    assert _coerce_rating(-3) == 0
    assert _coerce_rating("5") == 5
    assert _coerce_rating(None) == 0
