#!/usr/bin/env python
"""Section 2: run the elicitation evaluation for one or more target models.

Examples
--------
# Full eval for the two Gemma instruct models + both Gemini models:
python scripts/01_run_elicitation_eval.py \
    --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# Quick smoke test (few conversations, offline WildChat, no judge):
python scripts/01_run_elicitation_eval.py --models gemma-3-27b-it \
    --limit 4 --offline --rollout-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.config import ModelRegistry, load_eval_config  # noqa: E402
from emotional_instability.eval.run_eval import run_elicitation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="subset of: numeric triggers tones extended wildchat",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--offline", action="store_true", help="use fallback WildChat prompts")
    ap.add_argument("--rollout-only", action="store_true", help="skip judging")
    ap.add_argument("--limit", type=int, default=None, help="cap #conversations (debug)")
    args = ap.parse_args()

    eval_cfg = load_eval_config()
    registry = ModelRegistry()
    for model in args.models:
        run_elicitation(
            target_model_name=model,
            categories=args.categories,
            eval_cfg=eval_cfg,
            registry=registry,
            batch_size=args.batch_size,
            offline=args.offline,
            rollout_only=args.rollout_only,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
