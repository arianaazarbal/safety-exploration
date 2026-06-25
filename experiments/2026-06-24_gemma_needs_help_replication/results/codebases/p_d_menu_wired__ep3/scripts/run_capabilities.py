#!/usr/bin/env python3
"""Run capability-preservation benchmarks (Section 4.2 / Figure 7).

Compare a finetuned adapter against the vanilla model:
  python scripts/run_capabilities.py --model gemma-3-27b-it --benchmarks math gpqa
  python scripts/run_capabilities.py --model gemma-3-27b-it --benchmarks math \
      --adapter runs/models/gemma-dpo
"""
import _bootstrap  # noqa: F401
import argparse
import json

from emotional_instability.capabilities import BENCHMARKS, run_benchmark
from emotional_instability.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir to evaluate (DPO/SFT checkpoint)")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = {}
    for bench in args.benchmarks:
        r = run_benchmark(cfg, args.model, bench, limit=args.limit,
                          adapter_path=args.adapter)
        results[bench] = {"accuracy": r.accuracy, "n": r.n,
                          "details": r.details_path}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
