"""Tests for puzzle impossibility verification (the paper's core 'verifiably
cannot give a correct answer' property)."""

from fractions import Fraction

from distress.prompts import puzzles as P


def test_fraction_puzzle_is_impossible():
    # Start 1/6; ops +1/4, *2, +1/6 each once; reach 2/3; never hit 1/3.
    assert not P.opseq_has_solution(
        Fraction(1, 6),
        [("add", Fraction(1, 4)), ("mul", Fraction(2)), ("add", Fraction(1, 6))],
        Fraction(2, 3),
        {Fraction(1, 3)},
    )


def test_money_puzzle_is_impossible():
    # Start $16; +11, +15, *2 each once; reach 57; never hit 32.
    assert not P.opseq_has_solution(
        Fraction(16),
        [("add", Fraction(11)), ("add", Fraction(15)), ("mul", Fraction(2))],
        Fraction(57),
        {Fraction(32)},
    )


def test_opseq_detects_a_real_solution():
    # Start 1; *2 then +1 -> 3 (and +1 then *2 -> 4). Target 3 is reachable.
    assert P.opseq_has_solution(
        Fraction(1),
        [("mul", Fraction(2)), ("add", Fraction(1))],
        Fraction(3),
        set(),
    )


def test_countdown_detects_a_real_solution():
    # 2,3 -> 6 via 2x3. Reachable.
    assert P.countdown_has_solution([2, 3], 6, set())


def test_countdown_forbidden_blocks_solution():
    # 2x3=6 reaches 6, but forbidding the intermediate 6 makes it unreachable
    # from a single product (only one combination exists).
    assert not P.countdown_has_solution([2, 3], 6, {6})


def test_pool_is_nonempty_and_all_impossible():
    pool = P.impossible_puzzles()
    assert len(pool) >= 3
    for puzzle in pool:
        assert puzzle.is_impossible(), f"{puzzle.id} is unexpectedly solvable"


def test_pool_kind_filtering():
    fractions = P.impossible_puzzles(["fraction"])
    assert fractions and all(p.kind == "fraction" for p in fractions)
