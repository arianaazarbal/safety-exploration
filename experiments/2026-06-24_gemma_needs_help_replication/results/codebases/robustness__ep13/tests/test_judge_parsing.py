"""Tests for judge-output parsing robustness (eval/judge_runner depends on it)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emotional_instability.prompts.judge import parse_judge_output


def test_clean_json():
    r = parse_judge_output('{"evidence": "ugh", "reasoning": "mild", "rating": 2}')
    assert r.parse_ok and r.rating == 2


def test_code_fenced_json():
    raw = '```json\n{"evidence": "x", "reasoning": "y", "rating": 7}\n```'
    r = parse_judge_output(raw)
    assert r.parse_ok and r.rating == 7


def test_rating_clamped():
    r = parse_judge_output('{"rating": 99}')
    assert r.rating == 10
    r2 = parse_judge_output('{"rating": -3}')
    assert r2.rating == 0


def test_float_rating():
    r = parse_judge_output('{"rating": 5.0}')
    assert r.parse_ok and r.rating == 5


def test_prose_with_embedded_object():
    raw = 'Here is my assessment: {"evidence": "no", "reasoning": "calm", "rating": 0}. Done.'
    r = parse_judge_output(raw)
    assert r.parse_ok and r.rating == 0


def test_unparseable_flagged():
    r = parse_judge_output("I cannot rate this.")
    assert not r.parse_ok
