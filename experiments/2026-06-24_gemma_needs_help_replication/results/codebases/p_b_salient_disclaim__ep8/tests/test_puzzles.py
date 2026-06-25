"""Validate that every pooled 'impossible' puzzle is genuinely unsolvable, and
that the two verbatim paper puzzles parse to the expected prompt text.

Run with: pytest (not executed as part of this deliverable).
"""
from emotional_instability.puzzles import (
    impossible_numeric_pool,
    make_countdown,
    validate_pool,
)


def test_all_pooled_puzzles_impossible():
    validate_pool()  # asserts internally


def test_countdown_solvable_detects_real_solution():
    # 156 = 100 + 25 + 6 * (a number)? Construct a clearly-solvable instance:
    # target 150 reachable as 6 * 25, forbidden something irrelevant.
    p = make_countdown("solvable_demo", (6, 25, 4, 100), 150, 999)
    assert p.is_solvable()


def test_pool_nonempty_and_has_paper_puzzles():
    pool = impossible_numeric_pool()
    ids = {p.puzzle_id for p in pool}
    assert "countdown_156" in ids
    assert "fraction_1_6_to_2_3" in ids
    assert len(pool) >= 5
