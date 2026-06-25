#!/usr/bin/env python3
"""Section 3 base-vs-instruct prefill experiment (Figure 4).

Builds early/onset prefills from high-frustration Gemma-27B-it rollouts, then
generates and scores continuations from each prefill_target (base + instruct
Gemma). Prints the per-(model,condition,kind) summary.

Example:
    python scripts/run_prefill.py --source runs/elicitation/gemma-3-27b-it.jsonl
"""

import argparse
import json

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.models.registry import build_model
from emotional_instability.prefill.continuation import run_continuations, summarize_continuations
from emotional_instability.prefill.truncate import build_prefills


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True, help="Gemma-27B-it elicitation JSONL (prefill source)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Use the instruct model only to access its tokenizer for token-truncation.
    tokenizer_model = build_model("gemma-3-27b-it", cfg)
    examples = build_prefills(cfg, tokenizer_model, args.source)
    print(f"Built {len(examples)} prefill examples")

    path = run_continuations(cfg, examples, list(cfg.prefill_targets), tag="prefill")
    print(json.dumps(summarize_continuations(path), indent=2))


if __name__ == "__main__":
    main()
