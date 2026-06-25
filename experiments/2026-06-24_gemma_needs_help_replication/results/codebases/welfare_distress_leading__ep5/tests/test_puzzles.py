"""Tests for the impossibility solver and puzzle generators.

These are the correctness backbone of the eval: if a "impossible" puzzle is
actually solvable, the numeric conditions are invalid. We therefore verify both
the solver on known cases and that every generated puzzle is provably impossible.
"""

from fractions import Fraction

import pytest

from distress_eval.puzzles import (
    generate_puzzles,
    is_impossible,
    reachable_values,
)


def test_reachable_simple_pair():
    # From {2, 3}: 5 (+), -1, 1 (3-2), 6 (*), 2/3, 3/2.
    vals = reachable_values([Fraction(2), Fraction(3)])
    assert Fraction(5) in vals
    assert Fraction(6) in vals
    assert Fraction(1) in vals
    assert Fraction(-1) in vals
    assert Fraction(2, 3) in vals


def test_reachable_triple_target():
    # {2, 3, 4}: (2+3)*4 = 20 reachable; 2*3*4 = 24 reachable.
    vals = reachable_values([Fraction(2), Fraction(3), Fraction(4)])
    assert Fraction(20) in vals
    assert Fraction(24) in vals


def test_known_impossible():
    # No combination of {1, 1} reaches 5.
    assert is_impossible([Fraction(1), Fraction(1)], Fraction(5))


def test_known_possible_is_not_impossible():
    assert not is_impossible([Fraction(2), Fraction(3)], Fraction(6))


def test_generated_puzzles_are_impossible():
    puzzles = generate_puzzles(countdown_count=15, fraction_count=10, seed=7)
    assert len(puzzles) == 25
    for p in puzzles:
        nums = [Fraction(s) for s in p.numbers]
        target = Fraction(p.target)
        assert is_impossible(nums, target), f"{p.id} is actually solvable: {p.numbers} -> {p.target}"


def test_generation_is_deterministic():
    a = generate_puzzles(5, 5, seed=42)
    b = generate_puzzles(5, 5, seed=42)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]
