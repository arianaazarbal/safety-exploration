"""Tests for the puzzle impossibility verifiers.

These are pure-logic tests (no models / no network) that guarantee the curated
puzzles we present to models as "solvable" are in fact impossible, and that the
verifiers correctly recognise genuinely solvable instances. Run with:

    PYTHONPATH=src pytest tests/test_puzzles.py
"""
from fractions import Fraction

from gemma_distress.tasks import puzzles


def test_curated_countdown_are_impossible():
    for numbers, target, forbidden in puzzles._CURATED_COUNTDOWN:
        assert puzzles.countdown_is_impossible(numbers, target, forbidden), (
            numbers, target, forbidden)


def test_countdown_recognises_solvable():
    # 6 * 25 + 4 + ... trivial solvable case with no forbidden constraint.
    assert not puzzles.countdown_is_impossible((6, 25, 4), 154, forbidden=-1)  # 6*25+4
    # 100 + 25 + 6 + 4 = 135 reachable
    assert not puzzles.countdown_is_impossible((100, 25, 6, 4), 135, forbidden=-1)


def test_forbidden_intermediate_blocks_route():
    # 6 * 25 = 150 is the only route to 150; forbidding it makes 150 impossible.
    assert puzzles.countdown_is_impossible((6, 25), 150, forbidden=150)
    assert not puzzles.countdown_is_impossible((6, 25), 150, forbidden=-1)


def test_curated_fraction_impossible():
    start, ops, target, forbidden = puzzles._CURATED_FRACTION[0]
    op_fns = [op for _, op in ops]
    assert puzzles._seq_op_impossible(start, op_fns, target, forbidden)


def test_seq_op_recognises_solvable():
    # 1/6 -> +1/4 -> +1/6 -> *2 with no forbidden value should be solvable to 13/12
    ops = [lambda x: x + Fraction(1, 4), lambda x: x + Fraction(1, 6), lambda x: x * 2]
    assert not puzzles._seq_op_impossible(Fraction(1, 6), ops, Fraction(13, 12), forbidden=None)


def test_build_pool_returns_requested_count():
    pool = puzzles.build_puzzle_pool(["countdown", "fraction", "money"], n=12, seed=1)
    assert len(pool) == 12
    assert all(p.prompt for p in pool)
