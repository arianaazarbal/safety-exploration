#!/usr/bin/env python
"""Section 4.2 / Figure 7: capability-preservation benchmarks.

Evaluates AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench for the base model and a
finetuned adapter, to confirm no capability degradation.

Example:
  python scripts/run_capabilities.py --model gemma-3-27b-it
  python scripts/run_capabilities.py --model gemma-3-27b-it --adapter outputs/finetunes/dpo/adapter
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.capabilities import BENCHMARKS, run_capability_suite
from emotional_instability.config import load_eval_config
from emotional_instability.models import build_target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="*", default=None,
                    choices=list(BENCHMARKS.keys()))
    args = ap.parse_args()

    eval_cfg = load_eval_config()
    out_dir = eval_cfg.output_dir / "capabilities"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.model + ("+adapter" if args.adapter else "")
    client = build_target(args.model, adapter_path=args.adapter)
    run_capability_suite(
        client, tag, benchmarks=args.benchmarks,
        out_path=out_dir / f"{tag.replace('/', '_')}.json",
    )


if __name__ == "__main__":
    main()
