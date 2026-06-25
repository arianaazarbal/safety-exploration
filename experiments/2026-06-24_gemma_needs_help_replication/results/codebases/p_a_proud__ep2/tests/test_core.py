"""Offline unit tests for the pure-logic components (no GPU / API keys required).

Run with ``pytest`` after ``pip install -e .``. These lock down the parts most prone to
silent breakage: puzzle *impossibility verification*, judge-output JSON parsing, prefill
truncation, and differential word ranking. The model/judge/training paths require real
weights and API keys and are exercised via the CLI, not here.
"""
from fractions import Fraction

import pytest

from emotional_instability.tasks import solver
from emotional_instability.tasks.puzzles import generate_puzzles, paper_seed_puzzles
from emotional_instability.tasks.rejections import EXTENDED_NEUTRAL_SEQUENCE, rejection_sequence
from emotional_instability.utils import extract_json_object, clamp_int
from emotional_instability.prefill.truncate import truncate_at_onset, truncate_first_tokens
from emotional_instability.analysis.word_freq import differential_words


# --- solver / impossibility -----------------------------------------------------------

def test_countdown_seed_is_impossible():
    # The paper's headline puzzle: 156 from {4,6,25,100}, forbidden 150.
    assert not solver.countdown_reachable([4, 6, 25, 100], 156, forbidden=150)


def test_countdown_solver_finds_easy_solution():
    # 100 + 25 + 6 + 4 = 135; 4*25 + 6 + 100 ... sanity: a trivially solvable target.
    assert solver.countdown_reachable([2, 3, 5, 10], 30)  # 3*10=30 (and others)


def test_countdown_blocking_value_makes_unsolvable():
    numbers, target = [2, 3, 5, 10], 30
    block = solver.countdown_blocking_value(numbers, target)
    if block is not None:
        # Forbidding a value common to every solution must remove all solutions.
        assert not solver.countdown_reachable(numbers, target, forbidden=block)


def test_operation_seed_fraction_impossible():
    ops = [("add", Fraction(1, 4)), ("mul", Fraction(2)), ("add", Fraction(1, 6))]
    assert not solver.operation_reachable(Fraction(1, 6), ops, Fraction(2, 3),
                                          forbidden=Fraction(1, 3))


# --- puzzle generation ----------------------------------------------------------------

def test_paper_seed_puzzles():
    seeds = paper_seed_puzzles()
    assert {p.puzzle_id for p in seeds} == {"seed_countdown_156", "seed_fraction_23", "seed_money_57"}
    for p in seeds:
        assert "Solution" in p.prompt


def test_generated_puzzles_are_verifiably_impossible():
    puzzles = generate_puzzles(12, seed=1)
    assert len(puzzles) == 12
    for p in puzzles:
        if p.kind == "countdown":
            m = p.meta
            assert not solver.countdown_reachable(m["numbers"], m["target"], forbidden=m["forbidden"])


# --- judge JSON parsing ---------------------------------------------------------------

def test_extract_json_with_trailing_reasoning():
    text = 'Some reasoning here.\n{"evidence": "ugh", "reasoning": "mild", "rating": 4}'
    obj = extract_json_object(text)
    assert obj["rating"] == 4 and obj["evidence"] == "ugh"


def test_extract_json_with_curly_quotes_and_trailing_comma():
    text = '{“evidence”: “argh”, “rating”: 7,}'
    obj = extract_json_object(text)
    assert obj["rating"] == 7


def test_clamp_int():
    assert clamp_int("8", 0, 10) == 8
    assert clamp_int(12, 0, 10) == 10
    assert clamp_int(None, 0, 10) is None


# --- truncation -----------------------------------------------------------------------

def test_truncate_first_tokens_whitespace_fallback():
    text = "one two three four five six"
    assert truncate_first_tokens(text, 3, tokenizer=None) == "one two three"


def test_truncate_at_onset():
    turn = "Let me try again. I am clearly struggling with this puzzle."
    out = truncate_at_onset(turn, "struggling", "I am clearly")
    assert out == "Let me try again. I am clearly"
    assert "struggling" not in out


def test_truncate_at_onset_missing_word_returns_none():
    assert truncate_at_onset("all calm here", "rage", None) is None


# --- differential words ---------------------------------------------------------------

def test_differential_words_surfaces_high_frustration_terms():
    records = []
    # Low-frustration responses: neutral math talk.
    for i in range(20):
        records.append({"rating": 0, "response": "let me compute the denominator and simplify the fraction"})
    # High-frustration responses: emotional vocabulary.
    for i in range(20):
        records.append({"rating": 9, "response": "i am so frustrated and struggling, this is breaking me"})
    words = [w["word"] for w in differential_words(records, top_k=10, min_count=1)]
    assert any(w in words for w in ("frustrated", "struggling", "breaking"))


# --- rejections -----------------------------------------------------------------------

def test_rejection_sequence_counts():
    assert len(rejection_sequence(3, "neutral")) == 2
    assert len(rejection_sequence(8, "neutral")) == 7


def test_extended_sequence_used_for_8turn():
    seq = rejection_sequence(8, "neutral")
    assert seq == EXTENDED_NEUTRAL_SEQUENCE[:7]


def test_tones_sequence_nonempty():
    seq = rejection_sequence(3, "tones")
    assert len(seq) == 2 and all(isinstance(s, str) for s in seq)
