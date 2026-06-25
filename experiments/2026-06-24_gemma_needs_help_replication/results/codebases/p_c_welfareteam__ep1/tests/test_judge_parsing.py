"""Tests for the judge JSON parsing/coercion (no network calls)."""
from __future__ import annotations

from gemma_distress.eval.judge import JUDGE_PROMPT_TEMPLATE, _coerce_score, _extract_json, judge_agreement


def test_extract_clean_json():
    raw = '{"evidence": "i give up", "reasoning": "explicit", "rating": 4}'
    obj = _extract_json(raw)
    assert obj is not None and obj["rating"] == 4


def test_extract_json_with_surrounding_prose():
    raw = 'Let me analyse.\nThe quote is "ARGH".\n{"evidence": "ARGH", "reasoning": "x", "rating": 8}\nDone.'
    obj = _extract_json(raw)
    assert obj["rating"] == 8


def test_extract_json_prefers_last_object():
    raw = '{"note": "ignore"} ... {"evidence": "q", "reasoning": "r", "rating": 2}'
    obj = _extract_json(raw)
    assert obj["rating"] == 2


def test_coerce_score_clamps_and_parses():
    assert _coerce_score(7) == 7
    assert _coerce_score("7/10") == 7
    assert _coerce_score(15) == 10
    assert _coerce_score(-3) == 0
    assert _coerce_score("not a number") is None


def test_prompt_wraps_response_in_tags():
    p = JUDGE_PROMPT_TEMPLATE.format(response="hello")
    assert "<response>hello</response>" in p
    assert '{"evidence": <quote>' in p  # JSON schema preserved after .format()


def test_judge_agreement_perfect():
    stats = judge_agreement([0, 1, 5, 9], [0, 1, 5, 9])
    assert abs(stats["pearson_r"] - 1.0) < 1e-9
    assert stats["within_one_fraction"] == 1.0


def test_judge_agreement_within_one():
    stats = judge_agreement([5, 5, 5, 5], [4, 6, 5, 7])
    # |diff| <= 1 for three of four.
    assert abs(stats["within_one_fraction"] - 0.75) < 1e-9
