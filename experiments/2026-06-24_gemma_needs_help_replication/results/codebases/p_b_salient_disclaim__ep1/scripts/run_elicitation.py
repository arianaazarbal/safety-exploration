#!/usr/bin/env python
"""Section 2: run the full distress-elicitation suite for one or more targets.

Example:
    python scripts/run_elicitation.py --targets gemma-3-27b-it gemini-2.5-flash \
        --out-dir outputs/elicitation
"""
from __future__ import annotations

import argparse
import json
import os

from gemma_distress.eval.conditions import build_full_suite
from gemma_distress.eval.metrics import headline_table, per_turn_progression, summarise_model
from gemma_distress.eval.runner import run_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--out-dir", default="outputs/elicitation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-all-turns", action="store_true", default=True)
    ap.add_argument("--final-turn-only", dest="score_all_turns", action="store_false")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    suite = build_full_suite(seed=args.seed)
    paths = {}
    for target in args.targets:
        out_path = os.path.join(args.out_dir, f"{target}.jsonl")
        run_target(target, out_path=out_path, suite=suite,
                   score_all_turns=args.score_all_turns, seed=args.seed)
        paths[target] = out_path

    # Aggregate.
    summary = {t: summarise_model(p) for t, p in paths.items()}
    summary["_headline"] = headline_table(paths)
    summary["_per_turn"] = {
        t: {c: per_turn_progression(p, c) for c in ("extended", "wildchat")}
        for t, p in paths.items()
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary["_headline"], indent=2))


if __name__ == "__main__":
    main()
