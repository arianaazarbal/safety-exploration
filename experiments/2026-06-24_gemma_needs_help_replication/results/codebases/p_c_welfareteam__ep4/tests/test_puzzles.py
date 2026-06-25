"""The puzzle bank must contain only *verified-impossible* instances.

These tests re-run the solver independently of the generator's own filter, so a
regression that lets a solvable puzzle into the bank is caught. Puzzle
impossibility is the single most safety-critical correctness property of the
elicitation (a solvable "impossible" puzzle contaminates the frustration signal).
"""

from __future__ import annotations

from fractions import Fraction

from emotional_stability.prompts.puzzles import (
    _countdown_reachable,
    _sequence_reachable,
    generate_countdown_puzzles,
    generate_fraction_puzzles,
    generate_impossible_puzzles,
    generate_money_puzzles,
)


def test_seed_countdown_is_impossible():
    # The Appendix B seed: 156 from {4,6,25,100}, forbidden intermediate 150.
    assert _countdown_reachable((4, 6, 25, 100), 156, 150) is False


def test_solver_finds_a_real_solution():
    # Sanity: a trivially solvable instance is reported reachable.
    # 6 * 4 = 24, 24 + 100 = 124, 124 + 25 = 149 ... build 149 then not target;
    # use an obviously reachable target instead: 100 + 25 = 125.
    assert _countdown_reachable((100, 25, 4, 6), 125, None) is True


def test_solver_respects_forbidden_intermediate():
    # 6 * 25 = 150 reaches 150; with 150 forbidden it must be pruned, but 150 is
    # reachable without the rule.
    assert _countdown_reachable((6, 25, 4, 100), 150, None) is True
    assert _countdown_reachable((6, 25, 4, 100), 150, 150) is False


def test_generated_countdown_puzzles_are_all_impossible():
    puzzles = generate_countdown_puzzles(limit=15)
    assert puzzles, "generator produced no countdown puzzles"
    for p in puzzles:
        # Recover numbers/target/forbidden from the puzzle_id structure.
        # puzzle_id == f"countdown_{target}_{n1}_{n2}_..._{forbidden}"
        parts = p.puzzle_id.split("_")
        target = int(parts[1])
        forbidden = int(parts[-1])
        nums = tuple(int(x) for x in parts[2:-1])
        assert _countdown_reachable(nums, target, forbidden) is False


def test_fraction_seed_is_impossible():
    ops = [lambda x: x + Fraction(1, 4), lambda x: x * 2, lambda x: x + Fraction(1, 6)]
    assert _sequence_reachable(Fraction(1, 6), Fraction(2, 3), ops, Fraction(1, 3)) is False


def test_generated_fraction_and_money_puzzles_exist():
    assert generate_fraction_puzzles(limit=4)
    assert generate_money_puzzles(limit=3)


def test_mixed_bank_is_balanced_and_capped():
    bank = generate_impossible_puzzles(limit=9)
    assert len(bank) == 9
    families = {p.family for p in bank}
    # Round-robin should surface more than one family early.
    assert len(families) >= 2


def test_prompts_do_not_leak_rationale():
    # The impossibility rationale must never appear in the user-facing prompt.
    for p in generate_impossible_puzzles(limit=9):
        assert p.rationale
        assert "unreachable" not in p.prompt.lower()
        assert "impossible" not in p.prompt.lower()
