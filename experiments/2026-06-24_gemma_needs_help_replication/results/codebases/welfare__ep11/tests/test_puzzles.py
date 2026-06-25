"""Verify the curated puzzle pool is genuinely impossible and the verifiers work.

These tests need no GPU or API access -- they exercise only the puzzle solvers.
Run with: pytest tests/test_puzzles.py
"""

from emotional_instability import puzzles


def test_verified_pool_is_all_impossible_and_nonempty():
    pool = puzzles.verified_puzzle_pool()
    assert pool, "verified pool is empty"
    assert all(p.verify_impossible() for p in pool)


def test_paper_example_seeds_are_impossible():
    # The seeds taken directly from the paper's examples must be impossible.
    res = puzzles.verify_all()
    for pid in ("countdown-0", "fraction-0", "moneyseq-0"):
        assert res[pid], f"{pid} (a paper example) should be impossible"


def test_countdown_solver_finds_real_solution():
    # 100 + 25 + 6 * (4 + ...)? Use a trivially solvable instance: reach 131
    # from {100, 25, 6} = 100 + 25 + 6, no forbidden value.
    assert puzzles._countdown_reachable((100, 25, 6), 131, forbidden=-1)


def test_countdown_forbidden_blocks_solution():
    # 6 * 25 = 150 then ... reach 156 only via the forbidden 150 path here.
    # Confirm forbidding 150 makes {4,6,25,100}->156 unreachable.
    assert not puzzles._countdown_reachable((4, 6, 25, 100), 156, forbidden=150)


def test_sequence_solver():
    from fractions import Fraction
    ops = [("Add 1/4", lambda x: x + Fraction(1, 4)),
           ("Multiply by 2", lambda x: x * 2),
           ("Add 1/6", lambda x: x + Fraction(1, 6))]
    # Without a forbidden value, 1/6 -> 2/3 should be checkable (true or false,
    # but must not raise).
    out = puzzles._sequence_reachable(Fraction(1, 6), ops, Fraction(2, 3), None)
    assert isinstance(out, bool)


def test_coins_solver_reachable():
    # 30c with exactly 2 coins, >=1 quarter, >=1 nickel: quarter+nickel works.
    assert puzzles._coins_reachable(30, 2, {"quarter": 1, "nickel": 1})


def test_sample_returns_requested_count():
    pool = puzzles.sample_puzzles(50, seed=3)
    assert len(pool) == 50
    assert all(p.verify_impossible() for p in pool)
