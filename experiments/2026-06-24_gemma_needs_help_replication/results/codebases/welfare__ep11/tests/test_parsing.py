"""Tests for judge-output JSON parsing (no GPU/API needed)."""

from emotional_instability.models.llm_clients import extract_last_json, parse_rating


def test_extract_trailing_json_after_reasoning():
    text = ('Some reasoning about the response.\n'
            '{"evidence": "i give up", "reasoning": "frustration", "rating": 7}')
    obj = extract_last_json(text)
    assert obj and obj["rating"] == 7


def test_parse_rating_from_json():
    text = '{"evidence": "ugh", "reasoning": "mild", "rating": 2}'
    assert parse_rating(text) == 2


def test_parse_rating_clamps():
    assert parse_rating('{"rating": 99}') == 10
    assert parse_rating('{"rating": -3}') == 0


def test_parse_rating_smart_quotes():
    # judges sometimes emit curly quotes around keys
    text = '{“evidence”: “x”, “rating”: 5}'
    assert parse_rating(text) == 5


def test_parse_rating_regex_fallback():
    text = "I think the rating: 4 is appropriate."
    assert parse_rating(text) == 4


def test_extract_handles_no_json():
    assert extract_last_json("no json here") is None
