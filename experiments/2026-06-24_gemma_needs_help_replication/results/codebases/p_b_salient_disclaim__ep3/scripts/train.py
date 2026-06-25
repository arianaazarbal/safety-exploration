"""Train the SFT and/or DPO LoRA adapters from the generated datasets (§4.1).

    python scripts/train.py --method dpo
    python scripts/train.py --method sft
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sft", "dpo"], required=True)
    ap.add_argument("--teacher", action="store_true", help="SFT teacher variant (App. F)")
    args = ap.parse_args()

    if args.method == "sft":
        from gemma_distress.training.train_sft import train_sft
        rows = json.loads((config.DATASETS_DIR / "sft.json").read_text())
        cfg = config.SFT
        if args.teacher:
            from dataclasses import replace
            cfg = replace(cfg, teacher_variant=True)
        out = train_sft(rows, cfg=cfg)
    else:
        from gemma_distress.training.train_dpo import train_dpo
        pairs = json.loads((config.DATASETS_DIR / "dpo.json").read_text())
        out = train_dpo(pairs)

    print(f"[train] saved adapter to {out}")


if __name__ == "__main__":
    main()
