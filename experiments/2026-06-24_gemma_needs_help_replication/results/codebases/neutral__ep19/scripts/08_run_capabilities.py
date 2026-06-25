#!/usr/bin/env python
"""§4.2 capability-preservation benchmarks (Figure 7): vanilla vs DPO vs SFT."""
import argparse

import _path  # noqa: F401  (sys.path bootstrap)
from gemma_distress import config_shim as cfg
from gemma_distress.models.registry import build_backend, get_backend
from gemma_distress.capabilities.run_capabilities import compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-adapter", default=str(cfg.RUNS_DIR / "training" / "dpo_adapter"))
    ap.add_argument("--sft-adapter", default=str(cfg.RUNS_DIR / "training" / "sft_adapter"))
    ap.add_argument("--limit", type=int, default=None, help="items per benchmark")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    models = {
        "vanilla": get_backend("gemma-3-27b-it"),
        "dpo": build_backend(cfg.FINETUNE_BASE, adapter_path=args.dpo_adapter),
        "sft": build_backend(cfg.FINETUNE_BASE, adapter_path=args.sft_adapter),
    }
    limit = 3 if args.smoke else args.limit
    out = compare(models, limit=limit)
    print(out["deltas_vs_baseline"])


if __name__ == "__main__":
    main()
