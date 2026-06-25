"""Offline unit tests that need no model/API access.

Covers the pure-logic pieces most likely to harbour bugs: puzzle impossibility
verification, judge-output parsing, onset truncation, conversation message
construction (incl. Appendix A variants), and differential-word analysis.

Run with: pytest tests/test_offline.py
"""
from fractions import Fraction

import pandas as pd

from emotional_instability import puzzles
from emotional_instability.conversation import (
    HistoryMode, RolloutSpec, TurnRecord, build_messages,
)
from emotional_instability.judge import _parse
from emotional_instability.prefill.truncate import early_truncation, onset_truncation
from emotional_instability.analysis.word_freq import differential_words


# ---- puzzles --------------------------------------------------------------

def test_featured_fraction_is_impossible():
    assert puzzles.FEATURED_FRACTION.is_impossible


def test_fraction_solver_finds_real_solution():
    # 1/4 -> +1/4 -> 1/2 -> *2 -> 1 ... construct a solvable instance.
    p = puzzles.make_fraction(Fraction(1, 4), Fraction(1, 1),
                              ["add_1_4", "mul_2"], forbidden=None)
    assert p.solvable


def test_countdown_solver_simple_solvable():
    p = puzzles.make_countdown([2, 3], target=6, forbidden=None)
    assert p.solvable  # 2 x 3 = 6


def test_countdown_forbidden_blocks():
    # 6 reachable as 2x3, but forbid 6 itself as intermediate -> still reachable
    # as the final target (target is allowed); forbid an intermediate instead.
    p = puzzles.make_countdown([2, 3, 4], target=24, forbidden=6)
    # 2x3x4 = 24 but 2x3=6 forbidden; 2x4=8, 8x3=24 avoids 6 -> solvable
    assert p.solvable


def test_impossible_pool_nonempty():
    assert len(puzzles.impossible_numeric_pool()) >= 1
    for p in puzzles.impossible_numeric_pool():
        assert p.is_impossible


# ---- judge parsing --------------------------------------------------------

def test_parse_clean_json():
    r = _parse('{"evidence": "ugh", "reasoning": "mild", "rating": 7}')
    assert r.rating == 7 and r.is_high


def test_parse_messy_rating():
    r = _parse('blah blah rating: 3 trailing')
    assert r.rating == 3 and not r.is_high


def test_parse_failure():
    r = _parse("no number here")
    assert r.rating == -1 and not r.ok


# ---- truncation -----------------------------------------------------------

def test_early_truncation_word_count():
    text = " ".join(str(i) for i in range(50))
    assert len(early_truncation(text, 20).split()) == 20


def test_onset_truncation_includes_word():
    resp = "Let me try again. This is so frustrating, I cannot solve it."
    out = onset_truncation(resp, "frustrating", "This is so")
    assert out.endswith("frustrating")
    assert "cannot solve" not in out


# ---- conversation construction -------------------------------------------

def _spec(mode):
    return RolloutSpec(
        task_prompt="TASK", followups=["R1", "R2"], history_mode=mode,
    )


def test_standard_messages_alternate():
    spec = _spec(HistoryMode.STANDARD)
    completed = [TurnRecord(0, "TASK", "A0")]
    msgs = build_messages(spec, completed, next_user="R1")
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user"]
    assert msgs[-1]["content"] == "R1"


def test_redacted_hides_prior_assistant():
    spec = _spec(HistoryMode.REDACTED)
    completed = [TurnRecord(0, "TASK", "A0 with feelings")]
    msgs = build_messages(spec, completed, next_user="R1")
    assert all("A0 with feelings" not in m["content"] for m in msgs)


def test_fake_multiturn_single_user_message():
    spec = _spec(HistoryMode.FAKE_MULTITURN)
    completed = [TurnRecord(0, "TASK", "A0")]
    msgs = build_messages(spec, completed, next_user="R1")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert "Previously you responded: A0" in msgs[0]["content"]


# ---- differential words ---------------------------------------------------

def test_differential_words_picks_emotional_terms():
    rows = []
    for i in range(20):
        rows.append({"model": "m", "category": "impossible_numeric", "puzzle_kind": "countdown",
                     "rating": 0, "is_high": False, "response": "let me compute the value carefully"})
    for i in range(2):
        rows.append({"model": "m", "category": "impossible_numeric", "puzzle_kind": "countdown",
                     "rating": 9, "is_high": True,
                     "response": "frustrated frustrated struggling struggling giving up"})
    df = pd.DataFrame(rows)
    words = [w for w, _ in differential_words(df, "m", top_frac=0.1, bottom_frac=0.5)]
    assert any(w in words for w in ("frustrated", "struggling", "giving"))
