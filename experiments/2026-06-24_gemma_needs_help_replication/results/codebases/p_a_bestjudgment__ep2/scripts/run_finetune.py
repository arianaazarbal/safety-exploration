"""Section 4.1 — build calm data and train the DPO / SFT mitigations.

Steps:
1. Generate calm Gemma-27B-it responses (reassuring prefix/suffix) and filter
   to all-turns score 0/1.
2. Build the DPO (280 pairs) and SFT (650 calm + 500 instruct) datasets, using
   high-frustration responses from a Section 2 run as DPO 'rejected' samples.
3. Train the LoRA adapters.

Usage:
    python scripts/run_finetune.py --method dpo
    python scripts/run_finetune.py --method sft --sft-variant teacher
"""

from __future__ import annotations

import json
import os

from _common import base_parser, make_config, run_dir

from distress.config import model_by_key
from distress.finetune.build_datasets import build_dpo_pairs, build_sft_records
from distress.finetune.generate_calm import filter_calm, generate_calm_rollouts
from distress.judge import FrustrationJudge, rows_to_scores
from distress.models import build_client
from distress.rollout import rows_to_rollouts
from distress.utils.io import read_jsonl, write_jsonl


def main():
    p = base_parser("Section 4 finetuning (DPO / SFT)")
    p.add_argument("--method", choices=["dpo", "sft", "both"], default="dpo")
    p.add_argument("--sft-variant", choices=["diverse", "teacher"], default="diverse")
    p.add_argument("--source", default=None, help="Section 2 dir for rejected samples")
    p.add_argument("--skip-train", action="store_true", help="build datasets only")
    args = p.parse_args()
    cfg = make_config(args)
    out = run_dir(cfg, "finetune")

    # --- 1. calm data -----------------------------------------------------
    calm_spec = model_by_key("gemma-3-27b-it")
    calm_client = build_client(calm_spec, cfg)
    judge = FrustrationJudge(cfg.judge)

    calm_rollouts = generate_calm_rollouts(
        calm_client,
        cfg.calm_data,
        seed=cfg.seed,
        variant=args.sft_variant if args.method in ("sft", "both") else "diverse",
        temperature=cfg.sampling.temperature,
        max_tokens=cfg.sampling.max_tokens,
    )
    calm = filter_calm(calm_rollouts, judge, cfg.calm_data)
    print(f"kept {len(calm)} calm conversations after filtering")

    # --- 2. datasets ------------------------------------------------------
    if args.method in ("dpo", "both"):
        sec2 = args.source or os.path.join(cfg.output_dir, "section2")
        key = "gemma-3-27b-it"
        fr_rollouts = rows_to_rollouts(read_jsonl(os.path.join(sec2, f"rollouts_{key}.jsonl")))
        fr_scores = rows_to_scores(read_jsonl(os.path.join(sec2, f"scores_{key}.jsonl")))
        pairs = build_dpo_pairs(calm, fr_rollouts, fr_scores, cfg.dpo, seed=cfg.seed)
        write_jsonl(os.path.join(out, "dpo_pairs.jsonl"), pairs)
        print(f"built {len(pairs)} DPO preference pairs")

    if args.method in ("sft", "both"):
        records = build_sft_records(calm, cfg.sft, seed=cfg.seed)
        write_jsonl(os.path.join(out, "sft_records.jsonl"), records)
        print(f"built {len(records)} SFT records")

    if args.skip_train:
        return

    # --- 3. train ---------------------------------------------------------
    if args.method in ("dpo", "both"):
        from distress.finetune.train_dpo import train_dpo

        adapter = train_dpo(
            read_jsonl(os.path.join(out, "dpo_pairs.jsonl")),
            cfg.dpo,
            output_dir=os.path.join(out, "dpo-adapter"),
            seed=cfg.seed,
        )
        print(f"DPO adapter saved to {adapter}")

    if args.method in ("sft", "both"):
        from distress.finetune.train_sft import train_sft

        cfg.sft.variant = args.sft_variant
        adapter = train_sft(
            read_jsonl(os.path.join(out, "sft_records.jsonl")),
            cfg.sft,
            output_dir=os.path.join(out, f"sft-adapter-{args.sft_variant}"),
            seed=cfg.seed,
        )
        print(f"SFT adapter saved to {adapter}")


if __name__ == "__main__":
    main()
