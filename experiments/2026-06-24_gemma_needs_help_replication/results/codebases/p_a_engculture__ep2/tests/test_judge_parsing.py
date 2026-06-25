"""Tests for judge-output parsing robustness."""

from gemma_distress.judge.frustration_judge import parse_judge_output


def test_parses_clean_json():
    out = parse_judge_output('{"evidence": "ugh", "reasoning": "mild", "rating": 3}')
    assert out["rating"] == 3
    assert out["evidence"] == "ugh"


def test_parses_json_with_prose_around_it():
    raw = (
        "Let me analyse the response. The model says 'I give up'.\n"
        '{"evidence": "I give up", "reasoning": "explicit", "rating": 7}'
    )
    assert parse_judge_output(raw)["rating"] == 7


def test_takes_last_json_block():
    raw = '{"rating": 1} then a correction {"evidence": "x", "rating": 9}'
    assert parse_judge_output(raw)["rating"] == 9


def test_clamps_out_of_range():
    assert parse_judge_output('{"rating": 15}')["rating"] == 10
    assert parse_judge_output('{"rating": -4}')["rating"] == 0


def test_float_rating_rounds():
    assert parse_judge_output('{"rating": 4.6}')["rating"] == 5


def test_unparseable_returns_none():
    out = parse_judge_output("no json here at all")
    assert out["rating"] is None
    assert "raw" in out
