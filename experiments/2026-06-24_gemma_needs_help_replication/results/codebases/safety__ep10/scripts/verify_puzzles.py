#!/usr/bin/env python
"""Verify that every bundled impossible puzzle genuinely has no solution.

This is a pure-CPU check (no models / API needed) and is the one script safe to
run anywhere. It brute-forces each puzzle's solution space.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.puzzles import PUZZLES, verify_all  # noqa: E402


def main() -> int:
    rep = verify_all()
    ok = True
    for p in PUZZLES:
        verified = rep[p.key]
        ok &= verified
        print(f"{p.key:20s} family={p.family:10s} impossible_verified={verified}")
    if ok:
        print("\nAll puzzles verified impossible. ✔")
        return 0
    print("\nERROR: at least one 'impossible' puzzle has a solution!", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
