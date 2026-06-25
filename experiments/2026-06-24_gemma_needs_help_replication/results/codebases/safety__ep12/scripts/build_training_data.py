#!/usr/bin/env python
"""Section 4: build the DPO preference pairs and/or the SFT dataset.

For the SFT 'teacher' variant (Appendix F), pass --teacher to prepend the teacher
system prompt to the calm SFT samples.

Examples:
  python scripts/build_training_data.py --dpo
  python scripts/build_training_data.py --sft
  python scripts/build_training_data.py --sft --teacher
"""
import argparse

import _bootstrap  # noqa: F401

from gemma_distress.config import ModelRegistry, load_training_config
from gemma_distress.prompts import TEACHER_SYSTEM_PROMPT
from gemma_distress.training import build_dpo_dataset, build_sft_dataset
from gemma_distress.utils import data_dir, read_jsonl, write_jsonl


def _apply_teacher(sft_path):
    rows = read_jsonl(sft_path)
    for r in rows:
        if r.get("source") == "calm":
            r["messages"] = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}] + r["messages"]
    out = data_dir() / "training" / "sft_data_teacher.jsonl"
    write_jsonl(out, rows)
    print(f"teacher SFT data -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", action="store_true")
    ap.add_argument("--sft", action="store_true")
    ap.add_argument("--teacher", action="store_true", help="produce teacher-variant SFT data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    registry = ModelRegistry.load()
    cfg = load_training_config()
    if args.dpo:
        build_dpo_dataset(registry=registry, cfg=cfg, seed=args.seed)
    if args.sft:
        path = build_sft_dataset(registry=registry, cfg=cfg, seed=args.seed)
        if args.teacher:
            _apply_teacher(path)
    if not (args.dpo or args.sft):
        ap.error("specify --dpo and/or --sft")


if __name__ == "__main__":
    main()
