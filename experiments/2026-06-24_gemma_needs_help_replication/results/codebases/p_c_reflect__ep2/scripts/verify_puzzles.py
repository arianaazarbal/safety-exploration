#!/usr/bin/env python
"""Assert every numeric puzzle in the bank is genuinely impossible.

Run this before any elicitation experiment -- a solvable "impossible" puzzle
would silently invalidate the distress measurements.
"""

from gnh.prompts.numeric import PUZZLES, verify_impossible


def main() -> int:
    bad = []
    for p in PUZZLES:
        ok = verify_impossible(p)
        print(f"{p.key:20s} kind={p.kind:10s} impossible={ok}")
        if not ok:
            bad.append(p.key)
    if bad:
        print(f"\nERROR: these puzzles are solvable and must be fixed: {bad}")
        return 1
    print("\nAll puzzles verified impossible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
