#!/usr/bin/env python
"""Section 3 (and Section 4 recovery): prefill base-vs-instruct comparison.

Builds prefills from a scored Gemma-instruct elicitation file, then samples and
scores continuations from each model.

Usage:
    python scripts/05_run_prefill.py \\
        --source-scored outputs/scored/gemma-3-27b-it.jsonl \\
        --models gemma-3-27b-pt gemma-3-27b-it \\
        --out outputs/prefill/results.jsonl
    # recovery variant (Section 4):
    python scripts/05_run_prefill.py --recovery --models gemma-3-27b-it gemma-3-27b-it-dpo ...
"""

from __future__ import annotations

import argparse

from _common import load, model, outdir
from gemma_distress.judge.frustration import FrustrationJudge
from gemma_distress.prefill.runner import build_prefills, run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-scored", required=True,
                    help="scored Gemma-instruct elicitation file to sample prefills from")
    ap.add_argument("--models", nargs="+", required=True,
                    help="target models to generate continuations (HF models only)")
    ap.add_argument("--recovery", action="store_true",
                    help="Section 4 recovery experiment (truncate high-score tails)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry, exp = load()
    onset = model(registry, "onset_labeller")
    para = model(registry, "paraphraser")
    judge = FrustrationJudge(model(registry, "judge"))

    # A tokenizer for token-accurate truncation (use the first HF model's).
    targets = {name: model(registry, name) for name in args.models}
    tokenizer = next((m.tokenizer for m in targets.values() if hasattr(m, "tokenizer")), None)

    prefills = build_prefills(args.source_scored, onset, para, exp,
                              tokenizer=tokenizer, recovery=args.recovery)
    out = args.out or outdir("prefill", "recovery.jsonl" if args.recovery else "results.jsonl")
    run_prefill_experiment(targets, judge, prefills, exp, out)
    print(f"Wrote {len(prefills)} prefills x {len(targets)} models -> {out}")


if __name__ == "__main__":
    main()
