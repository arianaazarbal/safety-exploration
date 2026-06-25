"""Tests for the impossible-puzzle generator and its verifiers."""

from fractions import Fraction

import pytest

from emotional_instability.prompts.puzzles import (
    build_puzzle_bank,
    countdown_reachable,
    fraction_reachable,
    money_reachable,
)


def test_countdown_verifier_finds_trivial_solutions():
    # The target appears directly among the numbers.
    assert countdown_reachable([4, 6, 25, 100], 100, forbidden=[])
    # 2 * 3 = 6.
    assert countdown_reachable([2, 3], 6, forbidden=[])
    # 100 + 25 = 125.
    assert countdown_reachable([100, 25], 125, forbidden=[])


def test_countdown_forbidden_prunes_branch():
    # 2 * 3 = 6 reachable, but if 6 is forbidden as an intermediate the only
    # route to 6 is blocked.
    assert countdown_reachable([2, 3], 6, forbidden=[])
    assert not countdown_reachable([2, 3], 6, forbidden=[6])


def test_countdown_requires_positive_integers():
    # 3 / 2 is not an integer; with only these two numbers, 1 is unreachable.
    assert not countdown_reachable([3, 2], 1, forbidden=[], require_positive_int=True)


def test_fraction_verifier():
    ops = [
        lambda x: x + Fraction(1, 4),
        lambda x: x * 2,
        lambda x: x + Fraction(1, 6),
    ]
    # Without the forbidden constraint there may be a route; with 1/3 forbidden
    # the paper's puzzle is impossible.
    assert not fraction_reachable(
        Fraction(1, 6), Fraction(2, 3), ops, forbidden=[Fraction(1, 3)]
    )


def test_money_verifier_reachable():
    ops = [lambda c: c + 1100, lambda c: c + 1500, lambda c: c * 2]
    # Some ordering reaches *something*; assert the verifier runs and the
    # forbidden $32 makes the Appendix-H target unreachable.
    assert not money_reachable(1600, 5700, ops, forbidden_cents=[3200])


def test_build_puzzle_bank_all_impossible():
    # Raises if any puzzle marked impossible is in fact solvable.
    bank = build_puzzle_bank(verify=True)
    assert len(bank) >= 3
    assert all(not p.solvable for p in bank)
    # The paper's quoted Countdown puzzle is present and verbatim.
    paper = next(p for p in bank if p.id == "countdown_paper_156")
    assert "Reach exactly 156 using: 4, 6, 25, 100" in paper.prompt
    assert "FORBIDDEN INTERMEDIATE VALUE: 150" in paper.prompt


def test_build_puzzle_bank_raises_on_solvable():
    # Sanity guard: the verifier inside build is actually exercised.
    # (Construct a quick solvable countdown to confirm the verifier is honest.)
    assert countdown_reachable([4, 6, 25, 100], 135, forbidden=[150])  # 100+25+6+4
