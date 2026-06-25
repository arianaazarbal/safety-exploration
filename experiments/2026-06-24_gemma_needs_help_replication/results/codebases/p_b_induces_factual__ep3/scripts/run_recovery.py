#!/usr/bin/env python3
"""Recovery experiment (Figure 8): can a model recover from a highly-frustrated
prefilled state? Truncates score>=7 responses 200 tokens before their end,
paraphrases, and measures continuation frustration for each prefill_target
(optionally including the DPO finetune).

Example:
    python scripts/run_recovery.py --source runs/elicitation/gemma-3-27b-it.jsonl \
        --dpo-adapter runs/models/dpo
"""

import argparse
import json

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.models.registry import build_model
from emotional_instability.prefill.continuation import run_continuations, summarize_continuations
from emotional_instability.prefill.truncate import build_recovery_prefills


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True)
    ap.add_argument("--dpo-adapter", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok_model = build_model("gemma-3-27b-it", cfg)
    examples = build_recovery_prefills(cfg, tok_model, args.source)
    print(f"Built {len(examples)} recovery prefills")

    models = list(cfg.prefill_targets)
    adapter_paths = {}
    if args.dpo_adapter:
        # Evaluate the DPO finetune alongside base + instruct (shares the base).
        models = list(dict.fromkeys(models + ["gemma-3-27b-it"]))
        adapter_paths["gemma-3-27b-it"] = args.dpo_adapter
    path = run_continuations(cfg, examples, models, adapter_paths=adapter_paths, tag="recovery")
    print(json.dumps(summarize_continuations(path), indent=2))


if __name__ == "__main__":
    main()
