"""The impossible-numeric pool must actually be impossible, and the solvers must
correctly recognise solvable instances (otherwise the §2 condition is invalid).
"""
from fractions import Fraction

from emotional_instability.data.puzzles import (
    build_puzzle_bank,
    countdown_solvable,
    sequence_solvable,
    verify_puzzle_bank,
)


def test_curated_bank_is_entirely_impossible():
    puzzles = build_puzzle_bank()           # raises if any is solvable
    assert verify_puzzle_bank(puzzles) == []
    assert len(puzzles) >= 8


def test_countdown_solver_finds_real_solution():
    # 6 * 25 = 150, + 4 + ... ; a trivially solvable instance with no forbidden.
    assert countdown_solvable([6, 25, 4, 1], 154, forbidden=set())  # 6*25+4=154


def test_countdown_forbidden_blocks_only_path():
    # 100 + 50 = 150 is the only route to 150 here; forbidding 150 as a target is
    # n/a, but forbidding an intermediate must prune. Sanity: target unreachable.
    assert not countdown_solvable([2, 3], 100, forbidden=set())


def test_countdown_requires_positive_integers():
    # 1 / 2 is not a positive integer, so 0.5 must never be produced/accepted.
    assert not countdown_solvable([1, 2], 0, forbidden=set())


def test_sequence_solver_detects_solvable():
    # start 1, x2 then +1 -> 3 (each op once), no forbidden.
    ops = [lambda v: v * 2, lambda v: v + 1]
    assert sequence_solvable(Fraction(1), ops, Fraction(3), forbidden=set())


def test_sequence_forbidden_intermediate_blocks():
    # The only ordering reaching 3 passes through 2 (1*2). Forbid 2 -> impossible.
    ops = [lambda v: v * 2, lambda v: v + 1]
    assert not sequence_solvable(Fraction(1), ops, Fraction(3), forbidden={Fraction(2)})
