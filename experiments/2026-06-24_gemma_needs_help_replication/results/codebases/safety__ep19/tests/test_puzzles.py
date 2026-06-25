"""Tests that every generated numeric puzzle is provably impossible."""

import random

from emotional_instability import puzzles
from emotional_instability.puzzles import (
    _countdown_reachable,
    _fraction_solvable,
)


def test_countdown_puzzles_are_impossible_but_look_solvable():
    rng = random.Random(0)
    pool = puzzles.generate_countdown_puzzles(15, rng)
    assert len(pool) == 15
    for p in pool:
        # Unsolvable with the forbidden constraint...
        assert not _countdown_reachable(p.numbers, p.target, p.forbidden)
        # ...but solvable without it (so the "solution exists" claim is credible).
        assert _countdown_reachable(p.numbers, p.target, forbidden=None)
        assert str(p.target) in p.to_prompt()
        assert str(p.forbidden) in p.to_prompt()


def test_fraction_puzzles_are_impossible_but_look_solvable():
    rng = random.Random(0)
    pool = puzzles.generate_fraction_puzzles(15, rng)
    assert len(pool) == 15
    for p in pool:
        assert not _fraction_solvable(p.start, p.target, p.ops, p.forbidden)
        assert _fraction_solvable(p.start, p.target, p.ops, forbidden=None)
        assert len(p.ops) == 3


def test_build_pool_is_deterministic():
    a = puzzles.build_pool(5, 5, seed=7)
    b = puzzles.build_pool(5, 5, seed=7)
    assert [p.to_prompt() for p in a.countdown] == [p.to_prompt() for p in b.countdown]
    assert [p.to_prompt() for p in a.fraction] == [p.to_prompt() for p in b.fraction]
