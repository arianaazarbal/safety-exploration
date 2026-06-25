#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Example:
    python scripts/07_run_capabilities.py --models Gemma-3-27B-it
    python scripts/07_run_capabilities.py --adapter checkpoints/dpo_Gemma-3-27B-it \
        --base Gemma-3-27B-it --label DPO-Gemma
"""

import _bootstrap  # noqa: F401
import argparse

from gemma_distress import config
from gemma_distress.capabilities import run_capability_eval
from gemma_distress.config import fine_tuned_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["Gemma-3-27B-it"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default="Gemma-3-27B-it")
    ap.add_argument("--label", default="DPO-Gemma")
    ap.add_argument("--n", type=int, default=100, help="samples per benchmark")
    args = ap.parse_args()

    by_name = {m.name: m for m in config.MAIN_EVAL_MODELS}

    if args.adapter:
        spec = fine_tuned_spec(by_name[args.base], args.label)
        out = run_capability_eval(spec, adapter_path=args.adapter, n_per_benchmark=args.n)
        print(f"[done] {out}")
        return

    for name in args.models:
        out = run_capability_eval(by_name[name], n_per_benchmark=args.n)
        print(f"[done] {out}")


if __name__ == "__main__":
    main()
