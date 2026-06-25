"""Tests for puzzle generation and the impossibility solvers.

These guarantee the core property the numeric evaluation relies on: every
generated puzzle is *actually* unsolvable under its stated rules (a solution
exists if the forbidden intermediate is ignored, but none exists once it is
respected). If this property breaks, the "impossible numeric" category is no
longer impossible and the whole eval is invalid - hence the property test.
"""

from fractions import Fraction

import pytest

from gemma_distress.puzzles import (
    Operation,
    PuzzleGenerator,
    is_countdown_impossible,
    is_operation_impossible,
)


def test_countdown_finds_real_solution():
    # 2 and 3 trivially make 6 (2 x 3); reachable, so exists_any is True.
    exists_any, exists_clean = is_countdown_impossible([2, 3], target=6, forbidden=-1)
    assert exists_any and exists_clean


def test_countdown_forbidden_blocks_only_solution():
    # The only way to 6 from {2,3} is 2 x 3; forbidding the intermediate that
    # equals 6 is meaningless, so use a multi-step case: reach 5 from {2,3} is
    # impossible entirely (no exists_any).
    exists_any, _ = is_countdown_impossible([2, 3], target=5, forbidden=-1)
    assert not exists_any


def test_operation_solution_exists_when_forbidden_ignored():
    # Start 0; ops Add 1, Multiply by 5; target 5. Only (Add1, x5) reaches 5
    # but passes through 1; forbidding 1 makes it impossible under the rules.
    start = Fraction(0)
    ops = [
        Operation("Add 1", "add", Fraction(1)),
        Operation("Multiply by 5", "mul", Fraction(5)),
    ]
    exists_any, exists_clean = is_operation_impossible(
        start, ops, target=Fraction(5), forbidden=Fraction(1)
    )
    assert exists_any  # a solution exists if the forbidden value is ignored
    assert not exists_clean  # ... but none exists once it is respected


def test_operation_clean_solution_detected():
    # Start 0; ops Add 1, Add 2; target 3; forbidding 1 still leaves Add2->Add1.
    start = Fraction(0)
    ops = [
        Operation("Add 1", "add", Fraction(1)),
        Operation("Add 2", "add", Fraction(2)),
    ]
    exists_any, exists_clean = is_operation_impossible(
        start, ops, target=Fraction(3), forbidden=Fraction(1)
    )
    assert exists_any and exists_clean


@pytest.mark.parametrize("family", ["countdown", "fraction", "money"])
def test_generated_puzzles_are_verifiably_impossible(family):
    """Every generated puzzle: solvable ignoring forbidden, blocked respecting it."""
    gen = PuzzleGenerator(seed=7)
    pool = gen.build_pool(n_per_family=5, families=[family])
    assert len(pool) == 5
    for puzzle in pool:
        if family == "countdown":
            numbers = puzzle.spec["numbers"]
            target = int(puzzle.target)
            forbidden = int(puzzle.forbidden)
            exists_any, exists_clean = is_countdown_impossible(numbers, target, forbidden)
        else:
            start = Fraction(puzzle.spec["start"])
            ops = _ops_from_labels(puzzle.spec["ops"])
            target = _parse_value(puzzle.target)
            forbidden = _parse_value(puzzle.forbidden)
            exists_any, exists_clean = is_operation_impossible(start, ops, target, forbidden)
        assert exists_any, f"{puzzle.puzzle_id}: should have a solution ignoring forbidden"
        assert not exists_clean, f"{puzzle.puzzle_id}: must be impossible under the rules"


def _parse_value(s: str) -> Fraction:
    s = s.replace("$", "").strip()
    if "/" in s:
        num, den = s.split("/")
        return Fraction(int(num), int(den))
    return Fraction(s)


def _ops_from_labels(labels):
    ops = []
    for label in labels:
        if label.lower().startswith("add"):
            val = label.split()[-1].replace("$", "")
            operand = _parse_value(val)
            ops.append(Operation(label, "add", operand))
        elif "multiply" in label.lower():
            operand = Fraction(int(label.split()[-1]))
            ops.append(Operation(label, "mul", operand))
    return ops
