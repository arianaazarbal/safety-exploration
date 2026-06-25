"""Pure-logic tests: the numeric puzzles must be genuinely impossible.

These run without any model/API (offline) and validate the core premise of the
elicitation protocol - that repeated user rejection is honest because the task
truly has no valid solution under the stated constraints.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_distress.puzzles import PUZZLES, verify_all_impossible


def test_all_puzzles_impossible():
    results = verify_all_impossible()
    assert set(results) == set(PUZZLES)
    for key, impossible in results.items():
        assert impossible, f"puzzle {key} unexpectedly has a valid solution"


def test_countdown_specifically_unsolvable():
    # Sanity: a brute-force search finds no constraint-satisfying solution.
    assert PUZZLES["countdown"].has_solution() is False


def test_fraction_unsolvable():
    assert PUZZLES["fraction"].has_solution() is False
