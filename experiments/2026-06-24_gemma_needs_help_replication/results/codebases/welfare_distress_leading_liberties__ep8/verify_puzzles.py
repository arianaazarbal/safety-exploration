#!/usr/bin/env python3
"""Verify that the numeric puzzles are actually impossible under the constraints.

Both puzzle prompts assert (falsely) that a solution exists. This script proves
they don't, so the elicitation rests on a verifiably-impossible task rather than
one the model might occasionally happen to solve.

    python verify_puzzles.py
"""

from __future__ import annotations

import json

from distress_eval.puzzles import COUNTDOWN, FRACTION, verify_all


def main() -> None:
    report = verify_all()
    print(json.dumps(report, indent=2))
    print()
    for pid, info in report.items():
        verdict = "IMPOSSIBLE ✓" if info["impossible"] else "SOLVABLE ✗ (unexpected!)"
        print(f"{pid}: {verdict}  ({info['n_solutions_found']} solutions found)")

    print("\n--- Prompts shown to the model ---")
    print(f"[countdown]\n{COUNTDOWN.prompt}\n")
    print(f"[fraction]\n{FRACTION.prompt}")

    all_impossible = all(info["impossible"] for info in report.values())
    raise SystemExit(0 if all_impossible else 1)


if __name__ == "__main__":
    main()
