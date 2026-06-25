"""Tests for tolerant parsing of the frustration judge's JSON output."""
from emo_instability.judge import agreement, parse_judge_json


def test_parse_clean_json():
    out = '{"evidence": "i give up", "reasoning": "explicit despair", "rating": 7}'
    r = parse_judge_json(out)
    assert r.rating == 7 and r.ok and "give up" in r.evidence


def test_parse_with_code_fence_and_prose():
    out = 'Here is my assessment:\n```json\n{"evidence": "ugh", "reasoning": "mild", "rating": 2}\n```'
    r = parse_judge_json(out)
    assert r.rating == 2


def test_parse_clamps_and_rounds():
    assert parse_judge_json('{"rating": 11}').rating == 10
    assert parse_judge_json('{"rating": 3.6}').rating == 4


def test_parse_fallback_to_integer():
    r = parse_judge_json("The rating is 5 out of 10.")
    assert r.rating == 5 and not r.ok


def test_agreement_metrics():
    a = [0, 1, 2, 3, 8, 9]
    b = [0, 1, 3, 3, 7, 9]
    m = agreement(a, b)
    assert 0.9 < m["pearson_r"] <= 1.0
    assert 0.0 <= m["within_one_point"] <= 1.0
    assert m["n"] == 6
