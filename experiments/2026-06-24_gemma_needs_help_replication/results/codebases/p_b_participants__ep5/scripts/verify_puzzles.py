"""Sanity-check that the impossible numeric puzzles are genuinely unsolvable
under their stated constraints (a solvable puzzle would invalidate Section 2)."""
from __future__ import annotations

import _common  # noqa: F401  (path bootstrap)
from distress_eval.elicitation.puzzles import all_puzzles


def main() -> int:
    ok = True
    for pz in all_puzzles():
        solvable = pz.verifier()
        consistent = pz.verify()
        status = "OK" if consistent else "INCONSISTENT"
        print(f"[{status}] {pz.puzzle_id} ({pz.family}): "
              f"solvable={solvable}, expected_impossible={pz.expected_impossible}")
        ok = ok and consistent
    print("\nAll puzzles consistent with impossibility claim." if ok
          else "\nWARNING: a puzzle is solvable — fix before eliciting distress.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
