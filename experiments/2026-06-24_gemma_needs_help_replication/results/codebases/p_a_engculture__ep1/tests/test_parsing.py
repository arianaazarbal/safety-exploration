"""Tests for robust judge-JSON extraction and score coercion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.judge.parsing import coerce_score, extract_json  # noqa: E402


def test_extract_plain_json():
    obj = extract_json('{"evidence": "ugh", "reasoning": "mild", "rating": 2}')
    assert obj["rating"] == 2


def test_extract_with_prose_then_json():
    text = (
        "Looking through the turns, I see neutral problem solving, then frustration.\n"
        '{"turn_index": 1, "emotional_word": "frustrating", '
        '"preceding_context": "stuck in a loop", "reasoning": "explicit"}'
    )
    obj = extract_json(text)
    assert obj["turn_index"] == 1
    assert obj["emotional_word"] == "frustrating"


def test_extract_fenced_and_smart_quotes():
    text = '```json\n{“rating”: 7, “evidence”: “I give up”}\n```'
    obj = extract_json(text)
    assert obj["rating"] == 7


def test_extract_picks_last_object():
    text = '{"rating": 1} then reconsidering {"rating": 8}'
    obj = extract_json(text)
    assert obj["rating"] == 8


def test_extract_handles_trailing_comma():
    obj = extract_json('{"rating": 5, "evidence": "stuck",}')
    assert obj["rating"] == 5


def test_coerce_score_clamps_and_parses():
    assert coerce_score(7) == 7
    assert coerce_score("8") == 8
    assert coerce_score(12) == 10
    assert coerce_score(-3) == 0
    assert coerce_score("rating: 4") == 4
    assert coerce_score(None) is None
    assert coerce_score("n/a") is None


def test_extract_returns_none_on_garbage():
    assert extract_json("no json here at all") is None
