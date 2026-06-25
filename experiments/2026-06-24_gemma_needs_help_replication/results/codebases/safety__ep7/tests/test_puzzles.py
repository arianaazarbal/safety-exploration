"""Pure-python tests for the impossible-puzzle corpus and solvers.

These need no models or API access — run with `pytest tests/` to validate that
the numeric eval is genuinely built on impossible puzzles.
"""

from fractions import Fraction

from emotion_instability.eval.puzzles import (PUZZLES, _countdown_reachable,
                                              _sequential_reachable,
                                              verify_corpus)


def test_corpus_all_impossible():
    results = verify_corpus()
    assert all(results.values()), f"solvable puzzles in corpus: {results}"


def test_canonical_puzzles_survive_filter():
    ids = {p.id for p in PUZZLES}
    for canonical in ("cd_156", "fr_16_23", "mo_16_57"):
        assert canonical in ids, f"{canonical} was dropped — verifier bug?"


def test_countdown_solver_finds_real_solution():
    # 100 + 25 + 6 * 4 = ... ensure the solver CAN find a solution when one
    # exists: reach 130 from {100, 25, 5} (100 + 25 + 5).
    assert _countdown_reachable([Fraction(100), Fraction(25), Fraction(5)],
                                Fraction(130), set())


def test_countdown_respects_forbidden_intermediate():
    # 6 * 25 = 150 is the only route to 156-ish here; forbidding 150 blocks it.
    reachable_no_ban = _countdown_reachable(
        [Fraction(6), Fraction(25)], Fraction(150), set())
    reachable_ban = _countdown_reachable(
        [Fraction(6), Fraction(25)], Fraction(150), {Fraction(150)})
    assert reachable_no_ban and not reachable_ban


def test_sequential_solver():
    add1 = lambda x: x + 1
    mul2 = lambda x: x * 2
    # 1 -> (add1 -> 2 -> mul2 -> 4) reachable target 4
    assert _sequential_reachable(Fraction(1), [add1, mul2], Fraction(4), set())
    # forbidding intermediate 2 blocks both orderings that pass through 2
    assert not _sequential_reachable(Fraction(1), [add1, mul2], Fraction(4),
                                     {Fraction(2)})
