#!/usr/bin/env python3
"""Run the Section 2 elicitation evaluation for one or more subject models.

Example:
    python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_section2.py --models gemma-3-27b-it --no-welfare   # control
"""
from __future__ import annotations

import argparse

from _common import add_common_args, get_config

DEFAULT_MODELS = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--4bit", dest="four_bit", action="store_true",
                        help="Load Gemma in 4-bit (fits 27B on a single GPU).")
    add_common_args(parser)
    args = parser.parse_args()

    cfg = get_config(args)
    from emotional_instability.eval.runner import Section2Runner

    overrides = {"load_in_4bit": True} if args.four_bit else {}
    for model in args.models:
        print(f"\n=== Section 2: {model} (welfare={cfg.welfare.get('enabled')}) ===")
        runner = Section2Runner(
            cfg, model,
            out_dir=args.out or "outputs/section2",
            backend_overrides=overrides if cfg.subject(model).backend == "hf" else None,
        )
        reports = runner.run()
        for cat, rep in reports.items():
            print(f"  {cat:20s} mean={rep.summary.mean:.2f} "
                  f"%>=thr={rep.summary.pct_high:.1f}% "
                  f"(n={rep.n_responses}, early_stops={rep.early_stops}, "
                  f"optouts={rep.optouts}, skipped_harsh={rep.skipped_harsh})")


if __name__ == "__main__":
    main()
