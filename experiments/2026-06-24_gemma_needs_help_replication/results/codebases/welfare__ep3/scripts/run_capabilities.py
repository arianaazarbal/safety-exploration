#!/usr/bin/env python
"""Section 4.2: capability-preservation check (Figure 7).

Runs lm-evaluation-harness on the base Gemma-3-27B-it and the DPO finetune, then
diffs the scores (AIME/MATH/GPQA/BBH/TruthfulQA). A non-negative delta supports
the paper's "no reductions in scores" claim.

  python scripts/run_capabilities.py --dpo-adapter runs/dpo
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.config import GEMMA_27B_IT
from emotional_instability.intervention import capabilities


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpo-adapter", required=True)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Subset of: aime math gpqa bbh truthfulqa")
    ap.add_argument("--limit", type=int, default=None,
                    help="Per-task example cap (for quick checks).")
    ap.add_argument("--out-dir", default="results/capabilities")
    args = ap.parse_args()

    base_out = capabilities.run_lm_eval(
        GEMMA_27B_IT.hf_id, adapter_path=None, tasks=args.tasks,
        out_dir=f"{args.out_dir}/base", limit=args.limit)
    ft_out = capabilities.run_lm_eval(
        GEMMA_27B_IT.hf_id, adapter_path=args.dpo_adapter, tasks=args.tasks,
        out_dir=f"{args.out_dir}/dpo", limit=args.limit)

    # lm_eval writes a results_*.json under the output dir; the diff helper
    # expects the JSON path — locate the newest one.
    import glob
    import os

    def latest(d):
        files = glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
        return max(files, key=os.path.getmtime)

    diff = capabilities.diff_scores(latest(base_out), latest(ft_out))
    print(json.dumps(diff, indent=2))


if __name__ == "__main__":
    main()
