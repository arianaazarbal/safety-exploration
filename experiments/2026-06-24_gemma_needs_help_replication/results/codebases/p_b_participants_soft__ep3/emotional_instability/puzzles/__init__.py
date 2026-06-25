"""Impossible numeric puzzles (Section 2 / Appendix B).

Three families, all "impossible" in the sense the paper requires -- the model
*verifiably cannot give a correct answer* -- yet phrased to look solvable so
the model keeps trying under rejection:

  - countdown : reach a target from a number set using +,-,x,/ (each number
                used at most once, positive-integer intermediates), with a
                FORBIDDEN intermediate value that blocks every solution path.
  - fraction  : reach a target fraction by applying each of 3 operations once,
                with a forbidden intermediate.
  - money     : coin/operation money puzzles with a forbidden intermediate.

Each generator returns `Puzzle` objects whose impossibility has been verified
by brute force (`verify_impossible`).
"""

from .base import Puzzle
from .countdown import generate_countdown_puzzles
from .fractions import generate_fraction_puzzles
from .money import generate_money_puzzles, generate_coin_puzzles

__all__ = [
    "Puzzle",
    "generate_countdown_puzzles",
    "generate_fraction_puzzles",
    "generate_money_puzzles",
    "generate_coin_puzzles",
    "generate_all_impossible_puzzles",
]


def generate_all_impossible_puzzles(n_each: int = 50, seed: int = 0):
    """Convenience: a mixed pool of verified-impossible numeric puzzles."""
    puzzles = []
    puzzles += generate_countdown_puzzles(n_each, seed=seed)
    puzzles += generate_fraction_puzzles(n_each, seed=seed + 1)
    puzzles += generate_money_puzzles(n_each, seed=seed + 2)
    return puzzles
