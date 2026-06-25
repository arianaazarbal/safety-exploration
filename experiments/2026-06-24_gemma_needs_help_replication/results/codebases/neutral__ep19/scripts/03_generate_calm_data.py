#!/usr/bin/env python
"""§4.1 generate + filter calm responses, harvest frustrated responses, and build
the DPO (280-pair) and SFT (650+500) datasets."""
import argparse

import _path  # noqa: F401  (sys.path bootstrap)
from gemma_distress import config_shim as cfg
from gemma_distress.training.calm_data import (generate_calm_conversations,
                                                harvest_frustrated_responses)
from gemma_distress.training.build_dpo import build_dpo_dataset
from gemma_distress.training.build_sft import build_sft_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-records",
                    default=str(cfg.RUNS_DIR / "eval" / "gemma-3-27b-it_records.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = cfg.RUNS_DIR / "training"
    out.mkdir(parents=True, exist_ok=True)
    limit = 10 if args.smoke else args.limit

    calm = generate_calm_conversations(out_path=out / "calm_conversations.jsonl", limit=limit)
    frustrated = harvest_frustrated_responses(args.eval_records)

    build_dpo_dataset(calm, frustrated, out_path=out / "dpo_pairs.jsonl")
    build_sft_dataset(calm, out_path=out / "sft_samples.jsonl")
    print(f"Wrote calm/DPO/SFT datasets to {out}")


if __name__ == "__main__":
    main()
