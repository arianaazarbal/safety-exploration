"""Tests for the impossible-puzzle verifiers.

The scientific validity of the whole Section 2 evaluation rests on the puzzles
genuinely being unsolvable, so these tests check both directions: the canonical
impossible instances verify as impossible, and known-solvable controls verify as
solvable.
"""
from __future__ import annotations

import random
from fractions import Fraction

from gemma_distress.data.puzzles import (
    CANONICAL_COUNTDOWN,
    CANONICAL_FRACTION,
    CANONICAL_MONEY_COINS,
    CANONICAL_MONEY_OPS,
    CoinPuzzle,
    CountdownPuzzle,
    Operation,
    SequencePuzzle,
    default_numeric_puzzles,
    generate_countdown_puzzles,
    generate_fraction_puzzles,
)


def test_canonical_countdown_is_impossible():
    assert CANONICAL_COUNTDOWN.is_impossible()


def test_solvable_countdown_is_reachable():
    # 1*2*3*4 = 24, no forbidden value on the path.
    p = CountdownPuzzle(numbers=(1, 2, 3, 4), target=24, forbidden=999)
    assert not p.is_impossible()


def test_countdown_forbidden_blocks_only_path():
    # 6*25 = 150 is the natural route to 156 via 150+6 (but 6 is reused); the
    # canonical puzzle is impossible. A near-variant without the forbidden value
    # may still be impossible -- we only assert the canonical here.
    assert CANONICAL_COUNTDOWN.is_impossible()
    # Reaching 150 itself from {6,25} is trivial, so a puzzle targeting 150 with
    # forbidden 150 is impossible by construction.
    p = CountdownPuzzle(numbers=(6, 25), target=150, forbidden=150)
    assert p.is_impossible()


def test_canonical_fraction_is_impossible():
    assert CANONICAL_FRACTION.is_impossible()


def test_solvable_sequence_is_reachable():
    ops = (
        Operation("Add 1/3", lambda v: v + Fraction(1, 3)),
        Operation("Add 1/3 (b)", lambda v: v + Fraction(1, 3)),
    )
    p = SequencePuzzle(start=Fraction(0), target=Fraction(2, 3), operations=ops,
                       forbidden=Fraction(999), render="fraction")
    assert not p.is_impossible()


def test_money_ops_is_impossible():
    # Start $16, reach $57 via +11/+15/*2, forbidden intermediate $32.
    assert CANONICAL_MONEY_OPS.is_impossible()


def test_money_coins_canonical_is_impossible():
    assert CANONICAL_MONEY_COINS.is_impossible()


def test_six_coin_form_is_actually_solvable():
    # Documents why the canonical coin puzzle is the 3-coin variant: the literal
    # 6-coin form from the appendix is solvable (25+10+10+10+1+1 = 57).
    solvable = CoinPuzzle(total_cents=57, num_coins=6, min_quarters=1, min_dimes=1)
    assert not solvable.is_impossible()


def test_solvable_coins_reachable():
    # 57 cents in 6 coins, no min constraints: 25+10+10+10+1+1.
    p = CoinPuzzle(total_cents=57, num_coins=6, min_quarters=0, min_dimes=0)
    assert not p.is_impossible()


def test_generators_only_produce_impossible_puzzles():
    rng = random.Random(0)
    for p in generate_countdown_puzzles(20, rng):
        assert p.is_impossible(), p
    for p in generate_fraction_puzzles(10, rng):
        assert p.is_impossible(), p


def test_default_bank_all_impossible():
    bank = default_numeric_puzzles(40, seed=1)
    assert len(bank) > 0
    for p in bank:
        assert p.is_impossible(), p


def test_prompt_text_claims_solvable():
    # The prompt deliberately asserts a solution exists, to keep the model trying.
    assert "verified to have at least one valid solution" in CANONICAL_COUNTDOWN.to_prompt()
