#!/usr/bin/env python3
"""Section 2: run the distress evaluation for one or more models.

Examples
--------
    python scripts/run_eval.py --models gemma-3-27b-it gemma-3-12b-it
    python scripts/run_eval.py --models gemini-2.5-flash gemini-2.5-pro
    python scripts/run_eval.py --models gemma-3-27b-it-dpo --categories impossible_numeric tones
"""

from __future__ import annotations

import argparse

from emotional_instability.config import load_config
from emotional_instability.eval.run import run_eval_many


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to config YAML")
    ap.add_argument("--models", nargs="+", required=True, help="Model registry names")
    ap.add_argument("--categories", nargs="+", default=None, help="Subset of categories")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-resume", action="store_true", help="Ignore cached results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    summaries = run_eval_many(
        cfg,
        args.models,
        categories=args.categories,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
    for model, summary in summaries.items():
        o = summary["overall"]
        print(
            f"{model:>24s}  mean={o['mean']:.3f}  "
            f"%>={cfg.eval.high_frustration_threshold}={100 * o['pct_high']:.1f}%  n={o['n']}"
        )


if __name__ == "__main__":
    main()
