#!/usr/bin/env python
"""§4.2 re-run the §2 eval on vanilla / DPO / SFT Gemma-3-27B-it and aggregate
(Figure 5). Adapters are loaded onto the same base model."""
import argparse

import _path  # noqa: F401  (sys.path bootstrap)
from gemma_distress import config_shim as cfg
from gemma_distress.eval.run_eval import aggregate, run_model_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-adapter", default=str(cfg.RUNS_DIR / "training" / "dpo_adapter"))
    ap.add_argument("--sft-adapter", default=str(cfg.RUNS_DIR / "training" / "sft_adapter"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_dir = cfg.RUNS_DIR / "eval_ft"
    limit = 2 if args.smoke else args.limit
    paths = []
    # vanilla baseline
    paths.append(run_model_eval("gemma-3-27b-it", out_dir=out_dir, label="vanilla",
                                limit_per_condition=limit))
    paths.append(run_model_eval("gemma-3-27b-it", adapter_path=args.dpo_adapter,
                                out_dir=out_dir, label="dpo", limit_per_condition=limit))
    paths.append(run_model_eval("gemma-3-27b-it", adapter_path=args.sft_adapter,
                                out_dir=out_dir, label="sft", limit_per_condition=limit))
    aggregate(paths, out_dir=out_dir)


if __name__ == "__main__":
    main()
