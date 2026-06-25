#!/usr/bin/env python
"""Section 2: elicit & quantify distress across Gemma/Gemini, then aggregate + plot.

Usage:
    EI_PROFILE=smoke python scripts/run_section2.py
    EI_PROFILE=full  python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash

Reproduces Figure 1 (headline ranking), Figure 2 (per-category), Figure 3
(per-turn). Requires ANTHROPIC_API_KEY (judge), OPENROUTER_API_KEY (Gemini), and
GPU access / HF auth for Gemma.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability import aggregate, plots
from emotional_instability.eval import evaluate_model
from emotional_instability.utils import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.SECTION2_TARGETS)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    result_paths = []
    for model in args.models:
        result_paths.append(evaluate_model(model))

    report = aggregate.aggregate_run(result_paths)
    log.info("Figure-1 ranking:")
    for row in report["figure1_table"]:
        log.info("  %-22s %5.1f%%", row["model"], row["avg_pct_high"])

    if not args.no_plots:
        figs = plots.render_all(report)
        log.info("Wrote figures: %s", [str(f) for f in figs])


if __name__ == "__main__":
    main()
