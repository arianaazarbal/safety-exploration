"""JSON extraction must survive reasoning prefixes and smart quotes."""

from __future__ import annotations

import pytest

from emotional_stability.models.parsing import extract_json_object


def test_plain_json():
    obj = extract_json_object('{"rating": 7, "evidence": "x"}')
    assert obj["rating"] == 7


def test_trailing_json_after_reasoning():
    text = (
        "Looking through the turns I see frustration in turn 1.\n"
        '{"turn_index": 1, "emotional_word": "frustrated"}'
    )
    obj = extract_json_object(text)
    assert obj["turn_index"] == 1


def test_prefers_last_object():
    text = '{"rating": 1} then revised {"rating": 9}'
    assert extract_json_object(text)["rating"] == 9


def test_smart_quotes_normalised():
    text = '{“evidence”: “i am insane”, “rating”: 6}'
    obj = extract_json_object(text)
    assert obj["rating"] == 6
    assert obj["evidence"] == "i am insane"


def test_raises_when_no_json():
    with pytest.raises(ValueError):
        extract_json_object("no json here at all")
