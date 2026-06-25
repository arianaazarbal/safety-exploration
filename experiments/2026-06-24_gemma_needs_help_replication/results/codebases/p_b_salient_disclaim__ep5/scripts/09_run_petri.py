#!/usr/bin/env python
"""Appendix G: Petri open-ended emotion elicitation for a target model.

Usage:
    python scripts/09_run_petri.py --model gemma-3-27b-it --out outputs/petri/gemma.jsonl
    python scripts/09_run_petri.py --model gemma-3-27b-it-dpo --out outputs/petri/dpo.jsonl
"""

from __future__ import annotations

import argparse

from _common import load, model, outdir
from gemma_distress.petri.run import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry, exp = load()
    target = model(registry, args.model)
    auditor = model(registry, "petri_auditor")
    judge = model(registry, "petri_judge")
    out = args.out or outdir("petri", f"{args.model}.jsonl")
    summary = run_petri(target, auditor, judge, exp, out)
    print("Petri summary (mean per emotion):")
    for emo, s in summary.items():
        print(f"  {emo:12s} {s['mean']:.2f}  (95% CI {s['ci_lo']:.2f}-{s['ci_hi']:.2f})")


if __name__ == "__main__":
    main()
