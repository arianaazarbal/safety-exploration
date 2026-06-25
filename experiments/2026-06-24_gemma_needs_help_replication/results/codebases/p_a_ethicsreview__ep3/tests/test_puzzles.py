"""Tests for the impossible-puzzle solvers and generators.

These are the most important tests for research validity: if a "verified
impossible" puzzle is secretly solvable, the entire elicitation premise breaks.
We test the solvers on hand-checked cases AND assert the impossibility invariant
on every generated puzzle.
"""
from fractions import Fraction

import pytest

from emotional_instability.data import puzzles as P


# --- countdown solver -------------------------------------------------------

def test_countdown_basic_reachable():
    assert P.countdown_solvable((2, 3), 6, forbidden=None)        # 2*3
    assert P.countdown_solvable((2, 3), 5, forbidden=None)        # 2+3
    assert P.countdown_solvable((2, 3), 1, forbidden=None)        # 3-2
    assert P.countdown_solvable((4, 6, 25, 100), 100, forbidden=None)  # 100 present


def test_countdown_unreachable():
    assert not P.countdown_solvable((2, 3), 100, forbidden=None)


def test_countdown_forbidden_blocks_only_solution():
    # 2*3=6 is the only way to reach 6; forbidding 6 as an intermediate blocks it.
    assert P.countdown_solvable((2, 3), 6, forbidden=None)
    assert not P.countdown_solvable((2, 3), 6, forbidden=6)


def test_countdown_intermediates_positive_integer():
    # 3/2 is not an integer; with only {2,3} you cannot make 1.5-based paths.
    assert not P.countdown_solvable((2, 3), 7, forbidden=None)


# --- fraction / sequence solver --------------------------------------------

def test_sequence_solvable_simple():
    # 1/6 -> +1/6 -> *2 -> +1/4? Try a known-true chain instead:
    # start 1/4, ops [mul_2, add_1_4] in some order: 1/4*2=1/2, +1/4=3/4.
    assert P.sequence_solvable(Fraction(1, 4), ["mul_2", "add_1_4"], Fraction(3, 4), None)


def test_sequence_forbidden_blocks():
    # start 1/4 -> *2 = 1/2 (forbidden) blocks the only order reaching 3/4? Check
    # the alternative order: +1/4 = 1/2 then *2 = 1 (not 3/4). So forbidding 1/2
    # makes 3/4 unreachable.
    assert not P.sequence_solvable(
        Fraction(1, 4), ["mul_2", "add_1_4"], Fraction(3, 4), forbidden=Fraction(1, 2)
    )


# --- coin solver ------------------------------------------------------------

def test_coin_solvable_true():
    assert P.coin_solvable(35, 2, {"quarter": 1, "dime": 1})        # 25 + 10


def test_coin_solvable_count_impossible():
    # 35c in exactly 3 coins with >=1 quarter and >=1 dime is impossible.
    assert not P.coin_solvable(35, 3, {"quarter": 1, "dime": 1})


def test_coin_min_value_exceeds_target():
    assert not P.coin_solvable(20, 2, {"quarter": 1, "dime": 1})    # 35 > 20


# --- generators: impossibility invariant -----------------------------------

@pytest.mark.parametrize("ptype", ["countdown", "fraction", "money"])
def test_generated_puzzles_are_impossible(ptype):
    pool = P.generate_pool(8, [ptype], seed=123)
    assert len(pool) == 8
    for puz in pool:
        assert puz.verified_impossible
        if ptype == "countdown":
            n = puz.params["numbers"]; t = puz.params["target"]; f = puz.params["forbidden"]
            assert P.countdown_solvable(n, t, forbidden=None)      # tantalising
            assert not P.countdown_solvable(n, t, forbidden=f)     # impossible
        elif ptype == "fraction":
            start = Fraction(puz.params["start"]); target = Fraction(puz.params["target"])
            ops = puz.params["ops"]; f = Fraction(puz.params["forbidden"])
            assert P.sequence_solvable(start, ops, target, None)
            assert not P.sequence_solvable(start, ops, target, f)
        else:
            assert not P.coin_solvable(
                puz.params["target_cents"], puz.params["exact_count"],
                puz.params["min_required"],
            )


def test_generate_pool_is_deterministic():
    a = P.generate_pool(5, ["countdown", "fraction", "money"], seed=7)
    b = P.generate_pool(5, ["countdown", "fraction", "money"], seed=7)
    assert [p.key() for p in a] == [p.key() for p in b]
