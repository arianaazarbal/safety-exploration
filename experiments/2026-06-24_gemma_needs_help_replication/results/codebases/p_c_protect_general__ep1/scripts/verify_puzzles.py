#!/usr/bin/env python
"""Sanity check: confirm every shipped Countdown puzzle is actually impossible.

Run before sampling. A non-empty result means a puzzle is solvable and must be
replaced; otherwise the "verifiably impossible numeric task" claim fails.
"""

from __future__ import annotations

import sys

from emotional_eval.prompts.puzzles import BANK, verify_bank


def main() -> None:
    bad = verify_bank(BANK)
    print(f"Checked {len(BANK)} puzzles.")
    if bad:
        print(f"SOLVABLE (must replace): {bad}")
        sys.exit(1)
    print("All puzzles verified impossible.")


if __name__ == "__main__":
    main()
