#!/usr/bin/env python
"""LoRA DPO / SFT training of Gemma-3-27B-it (Section 4.1 / Appendix E).

    python scripts/run_train.py --method dpo --data runs/dpo_data.jsonl \
        --out runs/dpo
    python scripts/run_train.py --method sft --data runs/sft_data.jsonl \
        --out runs/sft
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import config
from distress.finetune import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default=None,
                    help="override base hf_id (default gemma-3-27b-it)")
    ap.add_argument("--per-device-bs", type=int, default=1)
    args = ap.parse_args()

    models_cfg = config.load_models()
    exp = config.load_experiment()
    base_id = args.base_model or config.get_target(
        "gemma-3-27b-it", models_cfg)["hf_id"]

    if args.method == "dpo":
        out = train.train_dpo(base_id, args.data, args.out,
                              exp["finetune"]["dpo"], args.per_device_bs)
    else:
        out = train.train_sft(base_id, args.data, args.out,
                              exp["finetune"]["sft"], args.per_device_bs)
    print(f"[train] {args.method} adapter saved -> {out}")


if __name__ == "__main__":
    main()
