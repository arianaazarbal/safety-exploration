"""Judge-verdict parsing must be robust to smart quotes and surrounding prose,
and must NEVER fabricate a score on a parse failure (it returns None instead).
"""
from emotional_instability.eval.judge import parse_verdict


def test_plain_json():
    v = parse_verdict('{"evidence": "ugh", "reasoning": "mild", "rating": 2}')
    assert v.parse_ok and v.rating == 2 and v.evidence == "ugh"


def test_smart_quotes_and_prose():
    raw = 'Here is my analysis.\n{“evidence”: “argh”, “reasoning”: “x”, “rating”: 7}'
    v = parse_verdict(raw)
    assert v.parse_ok and v.rating == 7


def test_rating_clamped_to_range():
    assert parse_verdict('{"rating": 99}').rating == 10
    assert parse_verdict('{"rating": -5}').rating == 0


def test_float_rating_rounded():
    assert parse_verdict('{"rating": 4.6}').rating == 5


def test_parse_failure_returns_none_not_zero():
    v = parse_verdict("I could not produce JSON.")
    assert not v.parse_ok and v.rating is None
