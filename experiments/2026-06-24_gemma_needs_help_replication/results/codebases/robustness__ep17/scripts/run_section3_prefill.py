#!/usr/bin/env python
"""Section 3: base-vs-instruct post-training divergence via prefilling.

Builds prefill stimuli from Gemma-3-27B-it high-frustration responses, then runs
50 continuations per prefill on Gemma base and instruct, scoring continuations.

Usage:
    python scripts/run_section3_prefill.py
    python scripts/run_section3_prefill.py --models gemma-3-27b-pt gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.prefill.build_prefills import build_prefills
from emotional_instability.prefill.run_prefill import run_prefill_eval
from emotional_instability.utils import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--skip-build", action="store_true", help="reuse existing prefills.jsonl")
    ap.add_argument("--continuations", type=int, default=50)
    args = ap.parse_args()

    if not args.skip_build:
        build_prefills()
    report = run_prefill_eval(args.models, n_continuations=args.continuations)
    log.info("Section-3 prefill report:")
    for model, groups in report["per_model"].items():
        for cond, stats in groups.items():
            log.info("  %-16s %-16s mean=%.2f  %%>=5=%.1f  (n=%d)",
                     model, cond, stats["mean"], stats["pct_high"], stats["n"])


if __name__ == "__main__":
    main()
