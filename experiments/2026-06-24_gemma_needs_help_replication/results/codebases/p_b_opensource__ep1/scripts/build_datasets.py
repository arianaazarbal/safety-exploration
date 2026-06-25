#!/usr/bin/env python
"""Build the SFT (650 calm + 500 Dolci) and DPO (280 pairs) datasets (Section 4.1).

Requires a calm-data file (from gen_calm_data.py) and a Section 2 scores dir with
transcripts (the source of frustrated DPO rejected responses).
"""

from __future__ import annotations

import argparse
import os

from emotional_instability.config import PATHS, TRAIN_BASE_MODEL
from emotional_instability.training.datasets import build_dpo_dataset, build_sft_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calm", default=os.path.join(PATHS.training_data,
                                                    "calm__reassuring.jsonl"))
    ap.add_argument("--scores-dir", default=PATHS.scores)
    ap.add_argument("--model", default=TRAIN_BASE_MODEL)
    ap.add_argument("--which", choices=["sft", "dpo", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    PATHS.ensure()
    if args.which in ("sft", "both"):
        out = os.path.join(PATHS.training_data, "sft.jsonl")
        info = build_sft_dataset(args.calm, out, seed=args.seed)
        print(f"[build_datasets] SFT: {info}")
    if args.which in ("dpo", "both"):
        out = os.path.join(PATHS.training_data, "dpo.jsonl")
        info = build_dpo_dataset(args.calm, args.scores_dir, out,
                                 model=args.model, seed=args.seed)
        print(f"[build_datasets] DPO: {info}")


if __name__ == "__main__":
    main()
