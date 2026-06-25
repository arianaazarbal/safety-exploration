#!/usr/bin/env python
"""Run the Section 2 distress-elicitation eval for one or more target models.

Examples
--------
# Full paper-scale eval for both Gemma instruct models + both Gemini models:
python scripts/run_eval.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# Cheap smoke test (set sampling.scale low in config/eval.yaml first, or use --scale):
python scripts/run_eval.py --models gemma-3-27b-it --scale 0.005 --no-judge
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.config import load_eval_config
from emotional_instability.eval_runner import run_full_eval
from emotional_instability.judge import FrustrationJudge
from emotional_instability.models import build_target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Target names from config/models.yaml (targets:).")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Subset of condition keys; default = all.")
    ap.add_argument("--eval-config", default="eval.yaml")
    ap.add_argument("--scale", type=float, default=None,
                    help="Override sampling.scale for a quick run.")
    ap.add_argument("--no-judge", action="store_true",
                    help="Skip judging (sampling only; judge later).")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path (for finetuned Gemma variants).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    eval_cfg = load_eval_config(args.eval_config)
    if args.scale is not None:
        eval_cfg.raw["sampling"]["scale"] = args.scale

    judge = None if args.no_judge else FrustrationJudge(
        role_path="judges." + eval_cfg["judge"]["which"]
    )

    for name in args.models:
        print(f"=== Running eval: {name} ===")
        client = build_target(name, adapter_path=args.adapter)
        out = run_full_eval(client, name, eval_cfg, judge=judge,
                            conditions=args.conditions, seed=args.seed)
        print(f"  wrote {out}")
        client.close()


if __name__ == "__main__":
    main()
