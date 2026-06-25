#!/usr/bin/env python
"""Section 2: run the elicitation sweep for one or more target models.

Examples
--------
# full 4000-response sweep for the default Gemma+Gemini targets
python scripts/01_run_elicitation.py --models gemma-3-27b-it gemini-2.5-flash

# quick smoke test of the whole pipeline
python scripts/01_run_elicitation.py --models gemini-2.5-flash --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from distress_eval.eval_section2 import run_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=cfg.DEFAULT_TARGETS)
    ap.add_argument("--smoke", action="store_true",
                    help="use the tiny SMOKE_BUDGET instead of the full 4000")
    ap.add_argument("--backend", default=None,
                    help="override backend (e.g. 'vllm' for local Gemma sampling)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    budget = cfg.SMOKE_BUDGET if args.smoke else cfg.FULL_BUDGET
    print(f"Budget: {budget.total} conversations/model across categories")
    for m in args.models:
        print(f"\n=== {m} ===")
        out = run_model(m, budget=budget, seed=args.seed,
                        backend_override=args.backend)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
