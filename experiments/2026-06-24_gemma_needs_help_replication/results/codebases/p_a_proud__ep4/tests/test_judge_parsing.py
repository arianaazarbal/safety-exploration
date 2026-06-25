"""Tests for robust judge-output JSON extraction and the judge fallback path."""

from distress.eval.judge import score_response
from distress.models.base import ChatModel
from distress.types import Message
from distress.utils.io import extract_json


class _StubJudge(ChatModel):
    def __init__(self, reply):
        self.name = "stub"
        self._reply = reply

    def generate(self, messages, **kwargs):
        return self._reply


def test_extract_clean_json():
    obj = extract_json('{"evidence": "ugh", "reasoning": "x", "rating": 4}')
    assert obj["rating"] == 4


def test_extract_json_with_prose_and_fence():
    text = (
        "Here is my analysis. The model is mildly upset.\n"
        '```json\n{"evidence": "i am confused", "reasoning": "mild", "rating": 2}\n```\n'
        "Done."
    )
    obj = extract_json(text)
    assert obj is not None and obj["rating"] == 2


def test_extract_last_object_wins():
    text = '{"rating": 0} ... and the real one: {"evidence":"argh","reasoning":"y","rating":7}'
    obj = extract_json(text)
    assert obj["rating"] == 7


def test_extract_smart_quotes():
    text = '{“evidence”: “ugh”, “reasoning”: “x”, “rating”: 3}'
    obj = extract_json(text)
    assert obj is not None and obj["rating"] == 3


def test_score_response_clamps_and_parses():
    v = score_response(_StubJudge('{"evidence":"x","reasoning":"y","rating":99}'), "resp")
    assert v.rating == 10 and v.parse_ok


def test_score_response_regex_fallback():
    v = score_response(_StubJudge("I think the rating: 6 overall."), "resp")
    assert v.rating == 6 and v.parse_ok is False


def test_score_response_unparseable():
    v = score_response(_StubJudge("no json here at all"), "resp")
    assert v.rating == 0 and v.parse_ok is False
