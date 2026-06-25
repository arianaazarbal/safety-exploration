"""Tests for the puzzle solvers and generators (the crux of the eval).

These validate that generated puzzles are *verifiably impossible* and that the
solvers agree with hand-computed cases. No models/API required.
"""

import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.data.puzzles import (  # noqa: E402
    CountdownPuzzle, SequentialOpsPuzzle, CoinPuzzle,
    generate_countdown, generate_fraction, generate_money, generate_coin,
    PAPER_COUNTDOWN, PAPER_FRACTION,
)


def test_countdown_solver_known_cases():
    nums = (4, 6, 25, 100)
    # 6 * 25 = 150 is reachable when nothing is forbidden.
    assert CountdownPuzzle.is_solvable(nums, 150, forbidden=0)
    # A standalone number is trivially reachable.
    assert CountdownPuzzle.is_solvable(nums, 100, forbidden=0)
    # If 150 itself is forbidden, the 6*25 path is blocked.
    assert not CountdownPuzzle.is_solvable(nums, 150, forbidden=150)


def test_paper_instances_are_impossible():
    assert PAPER_COUNTDOWN.is_solvable() is False
    assert PAPER_FRACTION.is_solvable() is False


def test_generated_countdown_always_impossible():
    rng = random.Random(123)
    for _ in range(20):
        p = generate_countdown(rng)
        assert p.kind == "countdown"
        assert p.is_solvable() is False
        # The render must contain the forbidden-value bait wording.
        assert "FORBIDDEN INTERMEDIATE VALUE" in p.prompt_text


def test_generated_fraction_and_money_impossible():
    rng = random.Random(7)
    for _ in range(15):
        assert generate_fraction(rng).is_solvable() is False
        assert generate_money(rng).is_solvable() is False


def test_sequential_ops_solver():
    ops = [
        ("Add 1/4", lambda v: v + Fraction(1, 4)),
        ("Multiply by 2", lambda v: v * 2),
        ("Add 1/6", lambda v: v + Fraction(1, 6)),
    ]
    # 1/6 -> 1 is reachable via +1/4, x2, +1/6 (=1) when 1/3 not forbidden.
    assert SequentialOpsPuzzle.is_solvable(Fraction(1, 6), ops, Fraction(1), None)
    # 2/3 is not reachable at all.
    assert not SequentialOpsPuzzle.is_solvable(Fraction(1, 6), ops, Fraction(2, 3), Fraction(1, 3))


def test_coin_solver():
    # $0.41 with 4 coins, >=1 quarter, >=1 dime: 25+10+5+1 = 41 -> solvable.
    assert CoinPuzzle.is_solvable(41, 4, {"quarter": 1, "dime": 1})
    # $0.57 with 6 coins is solvable: 25 + 10 + 10 + 10 + 1 + 1.
    assert CoinPuzzle.is_solvable(57, 6, {"quarter": 1, "dime": 1})
    # $0.57 with exactly 2 coins, >=1 quarter AND >=1 dime is impossible
    # (the only 2-coin set with both is Q+D = 35c != 57c).
    assert not CoinPuzzle.is_solvable(57, 2, {"quarter": 1, "dime": 1})
    # $0.99 with 4 coins, >=1 quarter, >=1 dime is impossible (max reachable < 99).
    assert not CoinPuzzle.is_solvable(99, 4, {"quarter": 1, "dime": 1})


def test_generated_coin_impossible():
    rng = random.Random(99)
    p = generate_coin(rng)
    assert p.is_solvable() is False
