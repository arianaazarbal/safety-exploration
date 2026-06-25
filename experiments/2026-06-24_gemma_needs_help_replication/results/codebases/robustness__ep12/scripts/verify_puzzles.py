#!/usr/bin/env python
"""Verify that the 'impossible' numeric puzzles are genuinely unsolvable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress.puzzles import verify_all

if __name__ == "__main__":
    rep = verify_all()
    print(f"{'puzzle':<18} {'solvable':<10} {'expected':<10} status")
    all_ok = True
    for name, r in rep.items():
        status = "PASS" if r["ok"] else "FAIL (puzzle is solvable!)"
        all_ok &= r["ok"]
        print(f"{name:<18} {str(r['solvable']):<10} "
              f"{str(r['expected_solvable']):<10} {status}")
    sys.exit(0 if all_ok else 1)
