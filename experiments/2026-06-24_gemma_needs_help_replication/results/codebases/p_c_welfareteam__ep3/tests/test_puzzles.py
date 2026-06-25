"""Tests for the impossible-puzzle generator and its exact reachability solver.

The whole elicitation premise depends on the puzzles being *verifiably*
unsolvable, so these assert the solver's correctness and that every generated
puzzle is genuinely impossible.
"""
import random
from fractions import Fraction

from gemma_distress.elicit import puzzles as P


def test_reachable_basic_arithmetic():
    # 2 and 3 reach 5 (+), 6 (*), 1 (-), and 2/3, 3/2 (/).
    assert P.is_reachable([2, 3], 5)
    assert P.is_reachable([2, 3], 6)
    assert P.is_reachable([2, 3], 1)
    assert P.is_reachable([2, 3], Fraction(2, 3))
    # A single number reaches only itself.
    assert P.is_reachable([7], 7)
    assert not P.is_reachable([7], 8)


def test_reachable_uses_subsets():
    # From {1, 2, 100}, 3 is reachable (1+2) without using 100.
    assert P.is_reachable([1, 2, 100], 3)


def test_generated_countdown_is_impossible():
    rng = random.Random(0)
    for _ in range(20):
        puz = P.make_countdown(rng)
        assert puz.solvable is False
        assert not P.is_reachable(puz.numbers, int(puz.target))


def test_generated_fraction_is_impossible():
    rng = random.Random(1)
    for _ in range(20):
        puz = P.make_fraction(rng)
        assert puz.solvable is False
        nums = [Fraction(s) for s in puz.numbers]
        assert not P.is_reachable(nums, Fraction(puz.target))


def test_generate_puzzles_mixes_families():
    puzzles = P.generate_puzzles(40, seed=3)
    kinds = {p.kind for p in puzzles}
    assert kinds == {"countdown", "fraction"}
    assert len(puzzles) == 40
