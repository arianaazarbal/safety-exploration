#!/usr/bin/env python
"""Capability-preservation benchmarks (§4.2, Figure 7).

    python scripts/run_capabilities.py --base gemma-3-27b-it
    python scripts/run_capabilities.py --base gemma-3-27b-it --adapter results/dpo/all

AIME and EmoBench are reported as 'needs_custom_task' (see DESIGN.md
§Capability benchmarks); the rest run through lm-eval.
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.config import ModelConfig, results_dir
from emotional_instability.interventions.capabilities import evaluate_capabilities

DEFAULT_TASKS = ["MATH", "GPQA", "BBH", "TruthfulQA", "AIME", "EmoBench"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="base model name (configs/models.yaml)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task")
    args = ap.parse_args()

    mcfg = ModelConfig()
    base_id = mcfg.get(args.base).hf_id
    out = results_dir() / "capabilities"
    res = evaluate_capabilities(base_id, args.adapter, args.tasks, out_dir=out, limit=args.limit)
    tag = args.adapter.replace("/", "_") if args.adapter else args.base
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
