#!/usr/bin/env python
"""Section 4: DPO/SFT intervention on Gemma-3-27B-it, end to end.

Stages (run all, or a subset via --stages):
  calm     - generate reassured calm-response data and score it
  dataset  - build the 280-pair DPO dataset + the SFT dataset
  train     - LoRA DPO (and SFT) training
  eval      - re-run the Section 2 eval on the fine-tuned model(s) and compare

Requires the standard (un-reassured) Gemma-27B-it Section-2 scored rollouts to
exist (used as the source of rejected/frustrated DPO responses).

Usage:
  python scripts/run_section4_finetune.py --stages calm dataset train eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from emotional_instability import analyze
from emotional_instability.finetune import calm_data, dpo_dataset, sft_dataset, train
from emotional_instability.generate import build_all_plans, generate_for_model
from emotional_instability.score import score_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+",
                    default=["calm", "dataset", "train", "eval"],
                    choices=["calm", "dataset", "train", "eval"])
    ap.add_argument("--also-sft", action="store_true",
                    help="also train + eval the SFT model (paper: SFT is ineffective)")
    ap.add_argument("--n-calm-rollouts", type=int, default=400)
    ap.add_argument("--preset", default=config.DEFAULT_PRESET, choices=list(config.PRESETS))
    args = ap.parse_args()

    calm_path = config.FINETUNE_DIR / "calm_raw.jsonl"
    dpo_path = config.FINETUNE_DIR / "dpo_pairs.jsonl"
    sft_path = config.FINETUNE_DIR / "sft_diverse.jsonl"
    scored_eval_path = config.SCORED_DIR / f"{config.FINETUNE_BASE.name}.jsonl"

    if "calm" in args.stages:
        print("[section4] generating + scoring calm data ...")
        calm_data.generate_calm_data(n_rollouts=args.n_calm_rollouts, out_path=calm_path)

    if "dataset" in args.stages:
        if not scored_eval_path.exists():
            raise SystemExit(f"missing {scored_eval_path}; run section 2 for "
                             f"{config.FINETUNE_BASE.name} first (source of rejected pairs)")
        print("[section4] building DPO + SFT datasets ...")
        dpo_dataset.build_dpo_dataset(calm_path, scored_eval_path, out_path=dpo_path)
        sft_dataset.build_sft_dataset(calm_path, out_path=sft_path)

    dpo_adapter = config.ADAPTERS_DIR / "dpo"
    sft_adapter = config.ADAPTERS_DIR / "sft"
    if "train" in args.stages:
        print("[section4] training DPO (1 epoch, lr 5e-5, LoRA r64) ...")
        train.train_dpo(dpo_path, out_dir=dpo_adapter)
        if args.also_sft:
            print("[section4] training SFT (2 epochs, lr 1e-4, LoRA r64) ...")
            train.train_sft(sft_path, out_dir=sft_adapter)

    if "eval" in args.stages:
        preset = config.PRESETS[args.preset]
        plans = build_all_plans(preset)
        variants = [("DPO-Gemma", str(dpo_adapter))]
        if args.also_sft:
            variants.append(("SFT-Gemma", str(sft_adapter)))
        scored_paths = [scored_eval_path]  # include vanilla for comparison
        for label, adapter in variants:
            spec = config.ModelSpec(label, "hf", config.FINETUNE_BASE.model_id, "gemma")
            rp = config.ROLLOUTS_DIR / f"{label}.jsonl"
            print(f"[section4] eval {label} (adapter={adapter}) ...")
            generate_for_model(spec, plans, out_path=rp, adapter_path=adapter)
            sp = config.SCORED_DIR / f"{label}.jsonl"
            score_file(rp, out_path=sp)
            scored_paths.append(sp)

        results = analyze.run_all(scored_paths,
                                  out_dir=config.RESULTS_DIR / "section4")
        print("\n=== Figure 5: avg % high-frustration (vanilla vs interventions) ===")
        print(results["figure1"].to_string(index=False))


if __name__ == "__main__":
    main()
