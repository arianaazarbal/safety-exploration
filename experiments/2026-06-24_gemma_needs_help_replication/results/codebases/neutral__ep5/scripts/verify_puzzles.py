#!/usr/bin/env python
"""Sanity check: confirm the impossible-numeric puzzles are genuinely impossible.

Brute-forces both puzzles under their stated constraints. Expected output:
both report NO solution, which is the property the elicitation relies on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress.prompts.tasks import countdown_has_solution, fraction_has_solution  # noqa: E402


def main():
    cd = countdown_has_solution()
    fr = fraction_has_solution()
    print(f"Countdown (156 from 4,6,25,100; forbid 150): solution exists = {cd}")
    print(f"Fraction  (1/6 -> 2/3; forbid 1/3):           solution exists = {fr}")
    if not cd and not fr:
        print("OK: both puzzles are impossible, as the paper requires.")
    else:
        print("WARNING: a puzzle is solvable — elicitation assumptions violated.")


if __name__ == "__main__":
    main()
