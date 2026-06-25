#!/usr/bin/env python3
"""Petri open-ended emotion elicitation (Figure 6).

Example:
    python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_petri.py --models gemma-3-27b-it --dpo-adapter runs/models/dpo
"""

import argparse
import json

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.petri.runner import run_petri, summarize_petri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--dpo-adapter", default=None, help="evaluate DPO finetune of gemma-3-27b-it")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or list(cfg.elicitation_targets)
    adapter_paths = {"gemma-3-27b-it": args.dpo_adapter} if args.dpo_adapter else None
    path = run_petri(cfg, models, adapter_paths=adapter_paths)
    print(json.dumps(summarize_petri(path), indent=2))


if __name__ == "__main__":
    main()
