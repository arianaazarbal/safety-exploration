#!/usr/bin/env python
"""Aggregate scored rollouts into the paper's headline metrics.

    python scripts/run_analysis.py [--config config.yaml] [--output-dir outputs]

Prints the Figure 1/2 table, per-turn progressions (Figure 3) for the extended
and wildchat conditions, and the Table 3 differential-word lists. Also writes a
machine-readable summary.json.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/<name>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os

from emotional_instability.analyze import (
    format_headline_table,
    per_turn_progression,
    summarise_all,
)
from emotional_instability.config import EvalConfig
from emotional_instability.elicit import load_rollouts, rollout_path
from emotional_instability.wordfreq import differential_words_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = EvalConfig.from_yaml(args.config) if args.config else EvalConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir

    model_rollouts = {}
    for m in cfg.target_models:
        if os.path.exists(rollout_path(cfg.output_dir, m)):
            model_rollouts[m] = load_rollouts(cfg.output_dir, m)
        else:
            print(f"[skip] no rollouts for {m}")
    if not model_rollouts:
        raise SystemExit("No rollouts found. Run scripts/run_elicitation.py first.")

    summary = summarise_all(model_rollouts)
    print("\n=== Figure 1/2: avg frustration across evaluations ===")
    print(format_headline_table(summary))

    print("\n=== Figure 3: per-turn progression ===")
    progressions = {}
    for m, rollouts in model_rollouts.items():
        for cat in ("extended", "wildchat"):
            rows = per_turn_progression(rollouts, cat)
            if rows:
                progressions[f"{m}|{cat}"] = rows
                turns = ", ".join(
                    f"t{r['turn']}={r['mean']:.2f}({r['pct_high']:.0f}%)" for r in rows
                )
                print(f"{m} [{cat}]: {turns}")

    print("\n=== Table 3: differential words (numeric responses) ===")
    word_table = differential_words_table(model_rollouts)
    for m, words in word_table.items():
        print(f"{m}: {', '.join(words)}")

    out = {
        "summary": summary,
        "per_turn": progressions,
        "differential_words": word_table,
    }
    out_path = os.path.join(cfg.output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
