#!/usr/bin/env python
"""Section 4.1: train the SFT or DPO LoRA intervention on Gemma-3-27B-it.

python scripts/train_intervention.py --method dpo --out adapters/dpo
python scripts/train_intervention.py --method sft --out adapters/sft

# Layer-range ablation (Section 4.2 "internal vs expressed"): adapters on layers 30-35 only
python scripts/train_intervention.py --method dpo --layers 30 35 --out adapters/dpo_l30-35
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

from dotenv import load_dotenv

from emotional_instability.config import DEFAULT, INTERVENTION_BASE_MODEL


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sft", "dpo"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    help="restrict LoRA to decoder layers [lo, hi] (ablation)")
    args = ap.parse_args()

    cfg = DEFAULT
    if args.layers:
        cfg = cfg.with_overrides(lora=replace(cfg.lora, layer_range=(args.layers[0], args.layers[1])))

    base = INTERVENTION_BASE_MODEL.model_id
    data_dir = os.path.join(cfg.data_dir, "calm")

    if args.method == "sft":
        from emotional_instability.interventions.train_sft import train_sft

        rows = _read(os.path.join(data_dir, "sft.jsonl"))
        # Split the combined file back into calm vs mix is unnecessary -- train on all rows.
        out = train_sft(base, rows, [], cfg, args.out)
    else:
        from emotional_instability.interventions.train_dpo import train_dpo

        pairs = _read(os.path.join(data_dir, "dpo.jsonl"))
        out = train_dpo(base, pairs, cfg, args.out)
    print(f"[train] {args.method} adapter saved to {out}")


if __name__ == "__main__":
    main()
