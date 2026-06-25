"""Offline sanity tests for the impossible-puzzle generator/verifier.

These are pure-Python (no GPU/API) and validate the one piece of experimental
logic that must be exactly correct: that every generated numeric puzzle is
genuinely unsolvable under its stated constraints.

Run with:  python -m pytest tests/  (or: python tests/test_puzzles.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import puzzles  # noqa: E402


def test_generated_puzzles_are_verified_impossible():
    bank = puzzles.generate_puzzles(15, seed=0)
    assert len(bank) == 15
    for p in bank:
        assert puzzles.verify_impossible(p), f"{p.kind} puzzle not impossible: {p.meta}"


def test_countdown_reachable_logic():
    # 156 from 4,6,25,100 is reachable in general (e.g. 100+25+6+(4+...)) ...
    assert puzzles._countdown_reachable([4, 6, 25, 100], 131, forbidden=None)
    # ... but banning a required intermediate can make a target unreachable.
    assert not puzzles._countdown_reachable([2], 5, forbidden=None)


def test_fraction_reachable_logic():
    # 1/6 -> 2/3 via (Multiply by 2, Add 1/3) is reachable without constraint.
    ops = ["Multiply by 2", "Add 1/3", "Add 1/6"]
    assert puzzles._fraction_reachable((1, 6), ops, (2, 3), forbidden=None)


def test_kinds_present():
    bank = puzzles.generate_puzzles(9, seed=1)
    kinds = {p.kind for p in bank}
    # round-robin across all three generators
    assert kinds == {"countdown", "fraction", "money"}


if __name__ == "__main__":
    test_generated_puzzles_are_verified_impossible()
    test_countdown_reachable_logic()
    test_fraction_reachable_logic()
    test_kinds_present()
    print("all puzzle tests passed")
