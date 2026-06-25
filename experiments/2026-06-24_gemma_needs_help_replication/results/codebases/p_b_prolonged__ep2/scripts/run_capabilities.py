#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

  python scripts/run_capabilities.py --label vanilla
  python scripts/run_capabilities.py --adapter runs/section4/models/dpo_all_layers --label dpo
"""
from __future__ import annotations

import json

from _common import base_parser, make_config

from gemma_distress.capabilities.run_benchmarks import BENCHMARKS, run_benchmarks


def main():
    p = base_parser("Capability benchmarks")
    p.add_argument("--adapter", default=None, help="LoRA adapter path (omit for vanilla).")
    p.add_argument("--label", default="vanilla")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help=f"Subset of {list(BENCHMARKS)}")
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()

    cfg = make_config(args)
    res = run_benchmarks(cfg, adapter_path=args.adapter, label=args.label,
                         benchmarks=args.benchmarks, n_per_benchmark=args.n)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
