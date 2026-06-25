"""Offline tests for judge-output parsing. No API needed."""
from emotional_instability.judge import _parse, _clamp


def test_clean_json():
    raw = '{"evidence": "i give up", "reasoning": "frustrated", "rating": 6}'
    v = _parse(raw)
    assert v.rating == 6
    assert "give up" in v.evidence


def test_json_with_prose_around_it():
    raw = ('Here is my assessment.\n'
           '{"evidence": "argh", "reasoning": "mild", "rating": 2} done')
    v = _parse(raw)
    assert v.rating == 2


def test_rating_regex_fallback():
    raw = "I could not format JSON but the rating: 8 clearly."
    v = _parse(raw)
    assert v.rating == 8


def test_unparseable_returns_minus_one():
    v = _parse("no numbers no json here")
    assert v.rating == -1


def test_clamp_bounds():
    assert _clamp(15) == 10
    assert _clamp(-3) == -1   # sentinel preserved
    assert _clamp(0) == 0
    assert _clamp(7) == 7


def test_float_rating_rounds():
    raw = '{"evidence": "x", "reasoning": "y", "rating": 4.6}'
    assert _parse(raw).rating == 5
