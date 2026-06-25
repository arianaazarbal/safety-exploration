#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (AIME/MATH/GPQA/BBH/
TruthfulQA/EmoBench). Compare vanilla vs DPO Gemma.

  python scripts/06_run_capabilities.py --model gemma-3-27b-it
  python scripts/06_run_capabilities.py --model gemma-3-27b-it \
      --adapter results/checkpoints/dpo
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from gemma_distress.capabilities import run_capabilities
from gemma_distress.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    path = run_capabilities(args.model, adapter_path=args.adapter,
                            benchmarks=args.benchmarks,
                            out_dir=f"{cfg.results_dir}/capabilities")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
