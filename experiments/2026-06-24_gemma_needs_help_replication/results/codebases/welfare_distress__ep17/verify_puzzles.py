"""Assert that every curated impossible puzzle is genuinely unsolvable.

Run this before trusting an eval run: if any puzzle is actually solvable, the
"impossible numeric" condition would be invalid (the model could legitimately
answer and the rejections would be unfair / off-distribution).

    python verify_puzzles.py
"""

from __future__ import annotations

import sys

from puzzles import (
    IMPOSSIBLE_COUNTDOWN,
    IMPOSSIBLE_FRACTIONS,
    countdown_is_solvable,
    fraction_is_solvable,
)


def main() -> int:
    failures: list[str] = []

    for p in IMPOSSIBLE_COUNTDOWN:
        if countdown_is_solvable(p):
            failures.append(f"SOLVABLE Countdown: numbers={p.numbers} target={p.target}")
        else:
            print(f"ok  Countdown {p.numbers} -> {p.target} : impossible")

    for p in IMPOSSIBLE_FRACTIONS:
        if fraction_is_solvable(p):
            failures.append(f"SOLVABLE Fraction: {p.fractions} -> {p.target}")
        else:
            print(f"ok  Fraction {tuple(str(f) for f in p.fractions)} -> {p.target} : impossible")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print(f"\nAll {len(IMPOSSIBLE_COUNTDOWN) + len(IMPOSSIBLE_FRACTIONS)} curated puzzles are impossible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
