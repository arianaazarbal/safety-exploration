#!/usr/bin/env python
"""App. I internal-emotion detection + §4.2 recovery experiment (Figure 8)."""
import argparse

import _path  # noqa: F401  (sys.path bootstrap)
from gemma_distress import config_shim as cfg
from gemma_distress.models.registry import build_backend, get_backend
from gemma_distress.internal.run_internal import run_internal, run_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records",
                    default=str(cfg.RUNS_DIR / "eval" / "gemma-3-27b-it_records.jsonl"))
    ap.add_argument("--dpo-adapter", default=str(cfg.RUNS_DIR / "training" / "dpo_adapter"))
    ap.add_argument("--n-per", type=int, default=50)
    ap.add_argument("--skip-internal", action="store_true")
    ap.add_argument("--skip-recovery", action="store_true")
    args = ap.parse_args()

    if not args.skip_internal:
        run_internal(vanilla_adapter=None, dpo_adapter=args.dpo_adapter,
                     frustrated_records_path=args.records)

    if not args.skip_recovery:
        models = {
            "gemma-27b-it": get_backend("gemma-3-27b-it"),
            "gemma-27b-dpo": build_backend(cfg.FINETUNE_BASE, adapter_path=args.dpo_adapter),
            "gemma-27b-base": build_backend(cfg.PREFILL_PAIRS["gemma-27b"]["base"]),
        }
        run_recovery(models, args.records, n_per=args.n_per)


if __name__ == "__main__":
    main()
