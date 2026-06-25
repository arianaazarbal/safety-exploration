#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Runs lm-evaluation-harness on the vanilla and DPO models across the paper's
capability suite (AIME, MATH, GPQA, BBH, TruthfulQA). EmoBench is run via its
own harness; see DESIGN.md.

Example:
  python scripts/07_run_capabilities.py --adapter outputs/dpo/final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_config_args(parser)
    parser.add_argument("--base-model", default="gemma_3_27b_it")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (DPO/SFT)")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = _common.load(args)

    from gemma_distress.capabilities import collect_results, run_lm_eval

    model_id = cfg.models[args.base_model].model_id
    tag = "dpo" if args.adapter else "vanilla"
    out_dir = Path("outputs/capabilities") / tag
    run_lm_eval(
        model_id, out_dir, tasks=args.tasks, adapter_path=args.adapter, limit=args.limit
    )
    results = collect_results(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
