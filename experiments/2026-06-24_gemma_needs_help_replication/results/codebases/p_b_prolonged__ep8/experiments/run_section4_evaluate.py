"""Section 4.2: re-run the Section 2.1 evaluations on vanilla / DPO / SFT Gemma.

Produces Figure 5 (mean frustration and % >= 5 across the Section 2.1
evaluations for each finetuned variant). The headline result: DPO drops the
average %-high-frustration from 35% to ~0.3%.

Usage:
    python experiments/run_section4_evaluate.py --phase both --load-in-4bit
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import config
from gemma_needs_help.analysis import aggregate
from gemma_needs_help.conditions import CONDITIONS
from gemma_needs_help.runner import generate_for_model, score_for_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["generate", "score", "analyse", "both"], default="both")
    ap.add_argument("--n-per-condition", type=int, default=config.RESPONSES_PER_CONDITION)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    targets = config.SECTION4_MODELS    # vanilla, DPO, SFT
    for target in targets:
        kw = {"load_in_4bit": args.load_in_4bit}
        if args.phase in ("generate", "both"):
            generate_for_model(target, CONDITIONS, n_per_condition=args.n_per_condition, **kw)
        if args.phase in ("score", "both"):
            score_for_model(target, CONDITIONS)

    if args.phase in ("analyse", "both"):
        path = aggregate.save_summary([t.name for t in targets])
        print("Figure 5 summary:", path)


if __name__ == "__main__":
    main()
