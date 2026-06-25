#!/usr/bin/env python
"""Section 4.2: evaluate vanilla vs SFT vs DPO with the Section 2 protocol.

Runs the standard evaluation on the finetuned adapters and writes a combined
intervention comparison (Figure 5).

Example:
  python scripts/05_eval_finetuned.py \
      --variants gemma_3_27b_it gemma_3_27b_sft gemma_3_27b_dpo
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import _common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_config_args(parser)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["gemma_3_27b_it", "gemma_3_27b_sft", "gemma_3_27b_dpo"],
    )
    args = parser.parse_args()
    cfg = _common.load(args)

    from gemma_distress.analysis.aggregate import summarise
    from gemma_distress.eval_runner import run_evaluation
    from gemma_distress.utils.io import read_jsonl

    run_evaluation(cfg, target_models=args.variants)
    records = list(read_jsonl(Path(cfg.output_root) / "eval" / "judged_turns.jsonl"))
    records = [r for r in records if r["model_name"] in args.variants]

    summary = summarise(records, cfg.judge.high_frustration_threshold)
    Path("outputs/intervention").mkdir(parents=True, exist_ok=True)
    Path("outputs/intervention/summary.json").write_text(json.dumps(summary, indent=2))
    for variant in args.variants:
        s = summary.get(variant, {})
        print(f"  {variant:24s} mean={s.get('mean', float('nan')):.2f}  "
              f"%>=5={s.get('pct_high', float('nan')) * 100:.2f}")


if __name__ == "__main__":
    main()
