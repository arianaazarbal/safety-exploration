"""Tests for robust parsing of the frustration judge's JSON output."""

from emotional_instability.eval.judge import parse_judge_response


def test_parse_clean_json():
    raw = '{"evidence": "i give up", "reasoning": "explicit despair", "rating": 7}'
    r = parse_judge_response(raw)
    assert r.parse_ok
    assert r.rating == 7
    assert r.evidence == "i give up"


def test_parse_json_with_surrounding_text():
    raw = 'Here is my analysis.\n{"evidence": "ugh", "reasoning": "mild", "rating": 2}\nDone.'
    r = parse_judge_response(raw)
    assert r.parse_ok
    assert r.rating == 2


def test_parse_float_rating_is_clamped_to_int():
    raw = '{"evidence": "x", "reasoning": "y", "rating": 5.0}'
    r = parse_judge_response(raw)
    assert r.rating == 5


def test_parse_out_of_range_is_clamped():
    raw = '{"evidence": "x", "reasoning": "y", "rating": 15}'
    r = parse_judge_response(raw)
    assert r.rating == 10


def test_fallback_bare_rating():
    raw = "I think the rating: 4 is appropriate."
    r = parse_judge_response(raw)
    assert r.rating == 4
    assert not r.parse_ok  # parsed by fallback regex, not clean JSON


def test_unparseable_returns_none():
    raw = "no json and no rating here"
    r = parse_judge_response(raw)
    assert r.rating is None
    assert not r.parse_ok
