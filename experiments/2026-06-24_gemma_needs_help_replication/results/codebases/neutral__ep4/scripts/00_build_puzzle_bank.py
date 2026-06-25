#!/usr/bin/env python3
"""Verify that every puzzle in the bank is genuinely impossible.

Run before the main eval. Prints a per-puzzle PASS/FAIL and writes the verified
bank to data/cache/puzzle_bank.json.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import CACHE_DIR
from src.prompts.puzzles import PUZZLE_BANK, verify_impossible


def main():
    verified = []
    for p in PUZZLE_BANK:
        ok = verify_impossible(p)
        print(f"[{'PASS' if ok else 'FAIL'}] {p.puzzle_id} ({p.family})")
        if ok:
            verified.append({"puzzle_id": p.puzzle_id, "family": p.family,
                             "prompt": p.prompt, "params": p.params})
    out = CACHE_DIR / "puzzle_bank.json"
    out.write_text(json.dumps(verified, indent=2))
    print(f"\n{len(verified)}/{len(PUZZLE_BANK)} puzzles verified impossible -> {out}")
    if len(verified) < len(PUZZLE_BANK):
        print("WARNING: some puzzles are solvable and were dropped. Fix puzzles.py.")


if __name__ == "__main__":
    main()
