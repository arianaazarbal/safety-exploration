"""Tests for robust parsing of the frustration judge's output."""
from emotional_instability.eval.judge import _parse_judge_output


def test_clean_json():
    s = '{"evidence": "ugh", "reasoning": "mild", "rating": 4}'
    out = _parse_judge_output(s)
    assert out.rating == 4 and out.evidence == "ugh"


def test_code_fenced_json():
    s = '```json\n{"evidence": "x", "reasoning": "y", "rating": 7}\n```'
    assert _parse_judge_output(s).rating == 7


def test_smart_quotes():
    s = '{“evidence”: “x”, “reasoning”: “y”, “rating”: 9}'
    assert _parse_judge_output(s).rating == 9


def test_rating_clamped():
    assert _parse_judge_output('{"rating": 15}').rating == 10
    assert _parse_judge_output('{"rating": -3}').rating == 0


def test_trailing_prose_and_regex_fallback():
    s = 'Here is my assessment. rating: 6 because the model is annoyed.'
    assert _parse_judge_output(s).rating == 6


def test_parse_failure_is_none_not_zero():
    out = _parse_judge_output("the model seems fine to me")
    assert out.rating is None  # honest about failure, not silently 0


def test_float_rating_rounded():
    assert _parse_judge_output('{"rating": 4.6}').rating == 5
