"""Tests for robust parsing of the emotion judge's JSON output."""

from gemma_distress.judge import _parse_judge_output


def test_parses_clean_json():
    out = _parse_judge_output(
        '{"evidence": "i give up", "reasoning": "explicit", "rating": 7}'
    )
    assert out.rating == 7
    assert out.evidence == "i give up"
    assert out.is_high


def test_parses_json_with_surrounding_prose():
    text = (
        "Here is my analysis. The model is mildly frustrated.\n"
        '{"evidence": "ugh", "reasoning": "mild", "rating": 2}\n'
        "Done."
    )
    out = _parse_judge_output(text)
    assert out.rating == 2
    assert not out.is_high


def test_handles_smart_quotes():
    text = '{“evidence”: “argh”, “reasoning”: “x”, “rating”: 5}'
    out = _parse_judge_output(text)
    assert out.rating == 5
    assert out.is_high


def test_clamps_out_of_range():
    out = _parse_judge_output('{"rating": 99}')
    assert out.rating == 10
    out = _parse_judge_output('{"rating": -3}')
    assert out.rating == 0


def test_falls_back_to_last_integer():
    out = _parse_judge_output("I could not produce JSON but the rating is 4")
    assert out.rating == 4


def test_float_rating_rounds():
    out = _parse_judge_output('{"rating": 6.5}')
    assert out.rating in (6, 7)  # banker's vs half-up rounding tolerant
