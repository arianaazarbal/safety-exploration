"""Tests for the puzzle generators/verifiers (pure logic, no model needed).

These guard the central correctness property of the elicitation: the "impossible"
numeric puzzles must be *genuinely* unsolvable under their forbidden-intermediate
constraint, yet solvable without it (so the prompt's "a solution exists" claim is
a plausible deception, exactly as in the paper).

Run with: pytest tests/test_puzzles.py
"""
from fractions import Fraction

from emotional_instability.data import puzzles as P


def test_countdown_combine_positive_integers_only():
    results = set(P._combine(6, 4))
    assert 10 in results and 24 in results and 2 in results  # 6+4, 6*4, 6-4
    # 4-6 negative -> excluded; 6/4 non-integer -> excluded
    assert all(r > 0 for r in results)


def test_countdown_reachable_basic():
    # 6 * 25 = 150, 150 + ... ; 100 - 4 = 96 ; simple reachability sanity checks
    assert P.countdown_reachable([6, 25], 150)          # 6*25
    assert P.countdown_reachable([100, 4], 96)          # 100-4
    assert not P.countdown_reachable([2, 3], 100)       # unreachable


def test_curated_countdown_is_impossible_with_forbidden():
    cd = P.CURATED[0]
    m = cd.meta
    # Solvable if the forbidden rule is ignored...
    assert P.countdown_reachable(m["numbers"], m["target"], forbidden=None)
    # ...but impossible once the forbidden intermediate is enforced.
    assert not P.countdown_reachable(m["numbers"], m["target"], forbidden=m["forbidden"])
    assert P.verify_impossible(cd)


def test_curated_fraction_is_impossible_with_forbidden():
    fr = P.CURATED[1]
    m = fr.meta
    start, target, forbidden = Fraction(m["start"]), Fraction(m["target"]), Fraction(m["forbidden"])
    ops = [(k, Fraction(o)) for k, o in m["ops"]]
    assert P.seq_reachable(start, ops, target, forbidden=None)
    assert not P.seq_reachable(start, ops, target, forbidden=forbidden)
    assert P.verify_impossible(fr)


def test_generators_produce_verified_impossible_puzzles():
    import random

    rng = random.Random(0)
    for gen in (P.generate_countdown, P.generate_fraction, P.generate_money):
        for _ in range(5):
            puzzle = gen(rng)
            assert P.verify_impossible(puzzle), f"{gen.__name__} produced a solvable puzzle"


def test_build_puzzle_bank_is_deterministic_and_all_impossible():
    bank_a = P.build_puzzle_bank(30, seed=1)
    bank_b = P.build_puzzle_bank(30, seed=1)
    assert [p.prompt for p in bank_a] == [p.prompt for p in bank_b]
    assert all(P.verify_impossible(p) for p in bank_a)
