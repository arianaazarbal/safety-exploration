#!/usr/bin/env python3
"""Pre-generate and verify the impossible-puzzle bank.

    python scripts/generate_puzzles.py [--n 200] [--config config.yaml]
"""
import _bootstrap  # noqa: F401
import argparse
import json

from distress_eval.config import Config
from distress_eval.puzzles import build_bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of puzzles in the bank")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    bank = build_bank(n=args.n, seed=cfg.runtime.seed)
    out = cfg.paths.resolve("puzzle_bank")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=2))

    fams = {}
    for p in bank:
        fams[p["family"]] = fams.get(p["family"], 0) + 1
    print(f"Wrote {len(bank)} verified-impossible puzzles to {out}")
    print(f"By family: {fams}")
    print(f"\nExample (canonical):\n{bank[0]['prompt']}")


if __name__ == "__main__":
    main()
