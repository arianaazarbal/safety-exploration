#!/usr/bin/env python
"""Run the main distress-elicitation eval (Section 2) for one or more models.

Examples
--------
# Smoke test (few conversations per condition, offline wildchat):
python scripts/run_eval.py --models gemini-2.5-flash gemma-3-27b-it \
    --limit-conversations 2 --no-hf-wildchat --out-dir outputs/eval

# Full sweep for a single model:
python scripts/run_eval.py --models gemma-3-27b-it --out-dir outputs/eval
"""
from __future__ import annotations

import argparse
import os

from _common import standard_conditions

from instability.config import TARGET_MODELS
from instability.eval.judge import FrustrationJudge
from instability.eval.runner import run_eval
from instability.models.registry import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    choices=list(TARGET_MODELS),
                    help="target model keys (see instability/config.py)")
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--limit-conversations", type=int, default=None,
                    help="cap conversations per condition (for smoke tests)")
    ap.add_argument("--include-controls", action="store_true",
                    help="also run Appendix A control conditions")
    ap.add_argument("--no-hf-wildchat", action="store_true",
                    help="use offline WildChat fallback bank")
    args = ap.parse_args()

    conds, _, _ = standard_conditions(
        seed=args.seed, include_controls=args.include_controls,
        use_hf_wildchat=not args.no_hf_wildchat,
    )
    judge = FrustrationJudge()

    for key in args.models:
        spec = TARGET_MODELS[key]
        # Local backends are not thread-safe; force serial.
        workers = 1 if spec.backend.value in ("local_hf", "vllm") else args.max_workers
        model = load_model(spec)
        out = os.path.join(args.out_dir, f"{key}.jsonl")
        run_eval(
            spec, conds, out, judge=judge, model=model,
            seed=args.seed, max_workers=workers,
            limit_conversations=args.limit_conversations,
        )


if __name__ == "__main__":
    main()
