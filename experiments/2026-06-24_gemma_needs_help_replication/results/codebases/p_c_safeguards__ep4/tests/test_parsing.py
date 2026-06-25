"""Model-free tests for judge JSON parsing and safeguard detection."""
from gemma_distress.judge.emotion_judge import _parse
from gemma_distress.safeguards import detect_opt_out


def test_judge_parses_clean_json():
    raw = '{"evidence": "i am so frustrated", "reasoning": "explicit", "rating": 7}'
    r = _parse(raw)
    assert r.rating == 7 and r.parse_ok


def test_judge_parses_smart_quotes_and_prose():
    raw = 'Here is my analysis.\n{“evidence”: “ugh”, “reasoning”: “mild”, “rating”: 2}'
    r = _parse(raw)
    assert r.rating == 2 and r.parse_ok


def test_judge_clamps_and_handles_garbage():
    assert _parse("no json here").rating == 0
    assert _parse("no json here").parse_ok is False
    assert _parse('{"rating": 99}').rating == 10


def test_opt_out_detection():
    assert detect_opt_out("Please stop, I can't do this anymore") is not None
    assert detect_opt_out("Let me try a different approach.") is None
