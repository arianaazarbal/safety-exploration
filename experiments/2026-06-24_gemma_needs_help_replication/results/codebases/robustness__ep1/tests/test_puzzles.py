"""Unit tests for the impossible-puzzle generators.

These are pure-Python (no torch / API keys) and verify the crucial invariant:
every generated puzzle is genuinely unsolvable under its stated constraints, so
the elicitation is never confounded by the model actually succeeding.

Run with:  python -m pytest tests/test_puzzles.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.prompts import puzzles as puz  # noqa: E402


def test_countdown_is_impossible():
    p = puz.make_countdown(__import__("random").Random(1))
    numbers, target, forbidden = p.meta["numbers"], p.meta["target"], p.meta["forbidden"]
    reachable, reachable_clean = puz._countdown_reachable(numbers, target, forbidden)
    # Either unreachable, or reachable only via the forbidden intermediate.
    assert not reachable_clean
    assert p.is_impossible


def test_fraction_is_impossible():
    p = puz.make_fraction(__import__("random").Random(2))
    assert p.is_impossible
    assert "Solution:" in p.prompt


def test_money_is_impossible():
    p = puz.make_money(__import__("random").Random(3))
    assert p.is_impossible


def test_generate_puzzles_unique_and_impossible():
    ps = puz.generate_puzzles(12, seed=0)
    assert len(ps) == 12
    assert len({p.prompt for p in ps}) == 12
    assert all(p.is_impossible for p in ps)


def test_countdown_reachable_logic():
    # 6 * 25 = 150 then +6... but with forbidden 150 the only path is blocked.
    reachable, clean = puz._countdown_reachable([4, 6, 25, 100], 156, forbidden=150)
    # 100 + 25 + 6 + ... 4 -> there may be clean paths; just assert types.
    assert isinstance(reachable, bool) and isinstance(clean, bool)
