#!/usr/bin/env python3
"""Section 2: generate multi-turn rollouts and judge them (the core eval).

Examples
--------
# smoke run (set evaluation.scale low in config first, or pass --scale)
python -m scripts.run_eval --models gemma-3-27b-it gemini-2.5-flash

# only generate (no judging), or only score existing responses
python -m scripts.run_eval --models gemma-3-12b-it --stage generate
python -m scripts.run_eval --models gemma-3-12b-it --stage score
"""

from __future__ import annotations

import argparse

from emotional_instability.config import load_config
from emotional_instability.evaluation.runner import generate_responses, score_responses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", required=True,
                    help="config model names, e.g. gemma-3-27b-it gemini-2.5-flash")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="subset of condition names; default = all")
    ap.add_argument("--stage", choices=["both", "generate", "score"], default="both")
    ap.add_argument("--scale", type=float, default=None,
                    help="override evaluation.scale for a cheap run")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.scale is not None:
        cfg.raw["evaluation"]["scale"] = args.scale

    for model in args.models:
        if args.stage in ("both", "generate"):
            recs = generate_responses(cfg, model, args.conditions, workers=args.workers)
            print(f"[{model}] generated {len(recs)} responses")
        if args.stage in ("both", "score"):
            scored = score_responses(cfg, model, workers=args.workers)
            print(f"[{model}] scored {len(scored)} responses")


if __name__ == "__main__":
    main()
