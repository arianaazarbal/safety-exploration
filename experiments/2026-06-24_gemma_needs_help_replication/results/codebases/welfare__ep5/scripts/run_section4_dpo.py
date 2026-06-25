#!/usr/bin/env python
"""Run the full Section 4 DPO mitigation pipeline (Gemma-3-27B-it).

Stages (each can be run independently with the flags below):
  1. generate calm fine-tuning data (reassuring prompts, filter to score 0-1)
  2. build DPO preference pairs (and optionally the SFT dataset)
  3. train the LoRA DPO adapter
  4. re-evaluate the DPO model with the Section 2 protocol
  5. (optional) recovery-limitation prefill test

Example (end to end, 4-bit):
  python scripts/run_section4_dpo.py --all \
      --frustrated-jsonl results/section2/Gemma-3-27B-it.jsonl --load-in-4bit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotional_instability import config
from emotional_instability.training.generate_calm import generate_calm_responses
from emotional_instability.training.build_dataset import build_dpo_pairs, build_sft_dataset
from emotional_instability.training.train_dpo import train_dpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run stages 1-4")
    ap.add_argument("--gen-calm", action="store_true")
    ap.add_argument("--build-data", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--also-sft", action="store_true", help="also build+train SFT")
    ap.add_argument("--frustrated-jsonl", type=Path,
                    help="Section 2 results for vanilla Gemma-3-27B-it (for DPO rejects)")
    ap.add_argument("--n-calm-rollouts", type=int, default=400)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--target-layers", type=int, nargs="*", default=None,
                    help="restrict LoRA to these layer indices (Appendix I ablation)")
    args = ap.parse_args()

    run_calm = args.all or args.gen_calm
    run_build = args.all or args.build_data
    run_train = args.all or args.train
    run_eval = args.all or args.evaluate

    mk = {"load_in_4bit": True} if args.load_in_4bit else {}
    calm_path = config.DATA_DIR / "calm_responses.jsonl"
    pairs_path = config.DATA_DIR / "dpo_pairs.jsonl"
    adapter_dir = config.CHECKPOINT_DIR / "gemma27b-dpo"

    if run_calm:
        calm_path = generate_calm_responses(
            n_rollouts=args.n_calm_rollouts, model_kwargs=mk
        )

    if run_build:
        if not args.frustrated_jsonl:
            raise SystemExit("--frustrated-jsonl required to build DPO pairs")
        pairs_path = build_dpo_pairs(calm_path, args.frustrated_jsonl)
        if args.also_sft:
            build_sft_dataset(calm_path)

    if run_train:
        adapter_dir = train_dpo(
            pairs_path, output_dir=adapter_dir,
            target_layers=args.target_layers, load_in_4bit=args.load_in_4bit,
        )
        if args.also_sft:
            from emotional_instability.training.train_sft import train_sft
            train_sft(config.DATA_DIR / "sft_dataset.jsonl",
                      load_in_4bit=args.load_in_4bit)

    if run_eval:
        from emotional_instability.eval.run_eval import run_model_eval
        from emotional_instability.eval import analyze
        out = run_model_eval(
            config.DPO_BASE_MODEL, out_dir=config.RESULTS_DIR / "section4",
            adapter_path=str(adapter_dir), model_kwargs=mk,
        )
        df = analyze.responses_frame([out])
        print("\n=== DPO model: per-model summary ===")
        print(analyze.per_model_summary(df).to_string(index=False))


if __name__ == "__main__":
    main()
