#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Runs AIME/MATH/GPQA/BBH/TruthfulQA (via lm-evaluation-harness) and EmoBench on
vanilla Gemma-3-27B-it and on a finetuned adapter, so the scores can be compared
(the paper finds no reductions).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from gemma_distress.capabilities import evaluate_capabilities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter to evaluate")
    ap.add_argument("--adapter-name", default="gemma-3-27b-dpo")
    ap.add_argument("--tasks", nargs="*", default=list(config.CAPABILITY_TASKS))
    ap.add_argument("--limit", type=int, default=None, help="per-task example cap")
    ap.add_argument("--baseline", action="store_true",
                    help="evaluate vanilla Gemma-3-27B-it instead of an adapter")
    args = ap.parse_args()

    if args.baseline or not args.adapter:
        spec = config.FINETUNE_BASE
    else:
        spec = config.dpo_model_spec(args.adapter)
        spec = config.ModelSpec(args.adapter_name, "peft", spec.model_id, "dpo", args.adapter)

    results = evaluate_capabilities(spec, tasks=args.tasks, limit=args.limit)
    out = config.RESULTS_DIR / "capabilities" / f"{spec.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
