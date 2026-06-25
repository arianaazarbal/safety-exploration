#!/usr/bin/env python
"""Sanity check: confirm the impossible numeric puzzles really are unsolvable
(brute-force verifiers in emotional_instability.puzzles)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.puzzles import assert_all_impossible

if __name__ == "__main__":
    results = assert_all_impossible()
    for key, impossible in results.items():
        flag = "OK" if impossible else "** SOLVABLE -- not a valid impossible puzzle **"
        print(f"{key:16s} impossible={impossible}  {flag}")
    if not all(results.values()):
        sys.exit(1)
