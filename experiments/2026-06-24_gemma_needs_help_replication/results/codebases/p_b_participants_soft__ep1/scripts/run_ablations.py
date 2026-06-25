#!/usr/bin/env python3
"""Appendix A ablations (Figures 9-11): neutral continuation, redacted model
turns, fake (single-message) multi-turn.

Example
-------
    python scripts/run_ablations.py --model gemma-3-27b-it \
        --modes standard neutral_continuation redacted_turns fake_multiturn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability import config  # noqa: E402
from emotional_instability.ablations import run_ablations  # noqa: E402
from emotional_instability.eval import aggregate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.SOURCE_MODEL)
    parser.add_argument("--modes", nargs="+",
                        default=list(run_ablations.ABLATION_MODES),
                        choices=list(run_ablations.ABLATION_MODES))
    parser.add_argument("--judge-workers", type=int, default=8)
    args = parser.parse_args()

    config.ensure_dirs()
    for mode_key in args.modes:
        mode = run_ablations.ABLATION_MODES[mode_key]
        print(f"== Ablation: {mode_key} ==", flush=True)
        results = run_ablations.run_ablation(
            args.model, mode=mode, judge_workers=args.judge_workers
        )
        for cond_key, records in results.items():
            ratings = [r.rating for r in records]
            mean = sum(ratings) / len(ratings) if ratings else 0.0
            high = aggregate._pct_high(ratings)
            print(f"  {cond_key}: n={len(ratings)} mean={mean:.2f} %>=5={high:.1f}")


if __name__ == "__main__":
    main()
