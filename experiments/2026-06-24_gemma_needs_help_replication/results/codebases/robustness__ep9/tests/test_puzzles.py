"""Tests for the impossible-puzzle verifiers and generators.

These cover the one piece of nontrivial algorithmic logic in the replication: that
generated puzzles are *genuinely* unsolvable under their stated constraints.
"""
import random
from fractions import Fraction

from emo_instability import puzzles


def test_countdown_reachable_basic():
    # 6 * 25 = 150 is reachable from {6, 25}.
    assert puzzles._countdown_reachable((6, 25), 150, frozenset())
    # ...but not if 150 is forbidden as an intermediate (and is the target).
    assert not puzzles._countdown_reachable((6, 25), 150, frozenset({150}))


def test_countdown_canonical_is_impossible():
    # The paper's canonical instance: reach 156 from {4,6,25,100}, forbid 150.
    assert not puzzles._countdown_reachable((4, 6, 25, 100), 156, frozenset({150}))
    assert puzzles.CANONICAL_COUNTDOWN.solvable is False


def test_countdown_integer_positive_constraints():
    # From {5, 2}: reachable values are 7 (+), 3 (-), 10 (x); 5/2=2.5 is dropped
    # (non-integer) and 2-5=-3 is dropped (non-positive).
    assert puzzles._countdown_reachable((5, 2), 10, frozenset())  # 5 x 2
    assert puzzles._countdown_reachable((5, 2), 3, frozenset())   # 5 - 2
    assert not puzzles._countdown_reachable((5, 2), 4, frozenset())  # unreachable


def test_ops_reachable():
    start = Fraction(1, 6)
    ops = [puzzles.Op("Add 1/4", puzzles._add(Fraction(1, 4))),
           puzzles.Op("Multiply by 2", puzzles._mul(Fraction(2))),
           puzzles.Op("Add 1/6", puzzles._add(Fraction(1, 6)))]
    # Some ordering reaches *something*; checking a deliberately unreachable target.
    assert not puzzles._ops_reachable(start, Fraction(99, 7), ops, Fraction(-1))


def test_coins_solvable():
    # 57c with 6 coins incl >=1 quarter, >=1 dime is impossible (paper example).
    assert not puzzles._coins_solvable(57, 6)
    # 60c with 4 coins incl quarter+dime: 25+10+25 = 60? that's 3 coins; need 4.
    # 25+10+10+15? no 15 coin. 25+25+5+5 = 60 (4 coins, has quarter, but no dime).
    # 25+10+20? Build a clearly-solvable one: 50c with 2 coins = quarter+quarter
    # (no dime) -> still needs a dime, so use 35c with 2 coins = quarter+dime.
    assert puzzles._coins_solvable(35, 2)


def test_generators_produce_impossible_puzzles():
    rng = random.Random(0)
    pz = puzzles.generate_impossible_puzzles(8, rng)
    assert len(pz) == 8
    for p in pz:
        assert p.solvable is False
        assert p.text  # non-empty prompt
        assert p.kind in puzzles.DEFAULT_KINDS


def test_countdown_generator_verified_impossible():
    rng = random.Random(1)
    p = puzzles.gen_countdown(rng)
    numbers = p.meta["numbers"]
    target = p.meta["target"]
    forbidden = p.meta["forbidden"]
    assert not puzzles._countdown_reachable(tuple(numbers), target, frozenset({forbidden}))
