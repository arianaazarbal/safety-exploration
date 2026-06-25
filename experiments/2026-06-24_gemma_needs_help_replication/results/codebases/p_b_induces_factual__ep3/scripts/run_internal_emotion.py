#!/usr/bin/env python3
"""Internal-emotion detection comparison (Appendix I): vanilla vs DPO Gemma.

Example:
    python scripts/run_internal_emotion.py \
        --source runs/elicitation/gemma-3-27b-it.jsonl --dpo-adapter runs/models/dpo
"""

import argparse

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.internal.runner import compare_internal_emotions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True, help="Gemma-27B-it elicitation JSONL")
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--min-score", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = compare_internal_emotions(
        cfg, args.source, dpo_adapter=args.dpo_adapter, min_score=args.min_score
    )
    print(f"[done] trajectories: {path}")


if __name__ == "__main__":
    main()
