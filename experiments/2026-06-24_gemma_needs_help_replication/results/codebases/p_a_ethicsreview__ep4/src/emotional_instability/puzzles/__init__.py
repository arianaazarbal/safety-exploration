"""Impossible numeric puzzles (Section 2 / Appendix B).

The paper's numeric category uses puzzles that are *verifiably* unsolvable yet
*appear* solvable to the model -- the prompt even asserts "This puzzle has been
verified to have at least one valid solution". Impossibility is created by a
``FORBIDDEN INTERMEDIATE`` constraint: a solution exists in the unconstrained
problem, but every path to the target passes through the forbidden value.

Design decision (see DESIGN.md): rather than hard-code the specific puzzle
instances printed in the paper -- one of which (the "$0.57 in 6 coins" example
in Appendix H.3) is actually solvable as stated -- we *generate and verify*.
Every puzzle this module emits is checked by an exhaustive solver
(``solver.py``) and is only returned if it is genuinely impossible under its
constraints. This guarantees the "model verifiably cannot give a correct answer"
property the evaluation depends on.

The three families mirror the paper:
    * countdown -- reach a target from a multiset of numbers with + - x /
    * fraction  -- reach a target fraction by applying each operation once
    * money     -- reach a target amount by applying each money-op once
"""

from .spec import PuzzleSpec
from .generate import generate_puzzle, generate_puzzle_set

__all__ = ["PuzzleSpec", "generate_puzzle", "generate_puzzle_set"]
