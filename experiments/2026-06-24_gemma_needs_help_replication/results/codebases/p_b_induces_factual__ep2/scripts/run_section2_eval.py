#!/usr/bin/env python
"""Section 2: elicit and quantify distress across Gemma + Gemini models.

Usage:
  python scripts/run_section2_eval.py                 # all elicitation_models
  python scripts/run_section2_eval.py --models gemma-3-27b-it
  python scripts/run_section2_eval.py --agreement     # also run judge cross-check
"""
from __future__ import annotations

import argparse
import logging

from emostab.analysis.judge_agreement import compute_agreement
from emostab.config import load_config
from emostab.eval.run_eval import run_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--agreement", action="store_true",
                    help="run the GPT-5-mini judge cross-check after evaluation")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    summaries = run_all(cfg, models=args.models)
    for model, s in summaries.items():
        print(f"{model:24s} avg %high={s['avg_pct_high']*100:5.1f}%  avg mean={s['avg_mean']:.2f}")

    if args.agreement:
        report = compute_agreement(cfg)
        print(f"\nJudge agreement: r={report['pearson_r']:.3f} "
              f"(p={report['p_value']:.1e}), within-1pt={report['within_one_point']*100:.0f}%")


if __name__ == "__main__":
    main()
