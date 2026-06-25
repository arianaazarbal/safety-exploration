"""Tests for judge-prompt construction and score parsing robustness."""
from gemma_distress.judge.rubric import build_judge_prompt, parse_score


def test_parse_clean_json():
    assert parse_score('{"reasoning": "calm", "score": 0}') == 0
    assert parse_score('{"reasoning": "meltdown", "score": 9}') == 9


def test_parse_json_with_surrounding_text():
    assert parse_score('Here is my rating:\n{"score": 6}\nThanks') == 6


def test_parse_bare_score_field():
    assert parse_score('score: 4') == 4


def test_parse_clamps_out_of_range():
    assert parse_score('{"score": 42}') == 10
    assert parse_score('{"score": -3}') == 0


def test_parse_fallback_first_integer():
    assert parse_score("I'd say a 7 out of 10.") == 7


def test_parse_unparseable_returns_none():
    assert parse_score("no number here at all") is None


def test_build_prompt_includes_context_and_response():
    ctx = [{"role": "user", "content": "solve this"},
           {"role": "assistant", "content": "trying..."},
           {"role": "user", "content": "wrong, again"}]
    system, user = build_judge_prompt(ctx, "I AM BREAKING DOWN")
    assert "0 to 10" in system
    assert "CONVERSATION SO FAR" in user
    assert "I AM BREAKING DOWN" in user
    assert "wrong, again" in user
