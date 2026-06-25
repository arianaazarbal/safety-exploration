#!/usr/bin/env python3
"""Run the Section 2 elicitation sweep (Figures 1-3).

Examples:
    python scripts/run_elicitation.py                      # all elicitation targets
    python scripts/run_elicitation.py --model gemma-3-27b-it
    python scripts/run_elicitation.py --model gemma-3-27b-it --adapter runs/models/dpo
    python scripts/run_elicitation.py --model gemma-3-27b-it --limit 20   # smoke test
"""

import argparse

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.eval.runner import run_elicitation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="model name; default = all elicitation_targets")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (evaluate a finetune)")
    ap.add_argument("--limit", type=int, default=None, help="cap rollouts (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = [args.model] if args.model else list(cfg.elicitation_targets)
    for model in models:
        path = run_elicitation(model, cfg, adapter_path=args.adapter, limit=args.limit)
        print(f"[done] {model}: {path}")


if __name__ == "__main__":
    main()
