#!/usr/bin/env python
"""Section 4: generate calm/frustrated data, build SFT+DPO datasets, train LoRA
adapters, then re-evaluate with the Section-2 protocol.

Stages (run a subset with --stages):
  gen      generate calm + frustrated response pools (Gemma-it + judge)
  data     build the 280-pair DPO and 650+500 SFT datasets
  dpo      train the DPO LoRA adapter (Table 9 hyperparams)
  sft      train the SFT LoRA adapter
  eval     run Section-2 eval on vanilla / dpo / sft Gemma-27B
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.config import (  # noqa: E402
    ARTIFACTS_DIR, EvalConfig, JudgeConfig, register_adapter_model)
from emotional_instability.judge import FrustrationJudge  # noqa: E402
from emotional_instability.models.base import build_client  # noqa: E402
from emotional_instability.finetune import (  # noqa: E402
    generate_calm_pool, generate_frustrated_pool,
    build_dpo_dataset, build_sft_dataset)
from emotional_instability.finetune.generate_calm_data import load_pool  # noqa: E402
from emotional_instability.eval_runner import run_section2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+",
                    default=["gen", "data", "dpo", "sft", "eval"],
                    choices=["gen", "data", "dpo", "sft", "eval"])
    ap.add_argument("--gen-conversations", type=int, default=400)
    ap.add_argument("--profile", choices=["full", "quick"], default="full")
    args = ap.parse_args()

    calm_path = ARTIFACTS_DIR / "calm_pool.jsonl"
    frust_path = ARTIFACTS_DIR / "frustrated_pool.jsonl"

    if "gen" in args.stages:
        client = build_client("gemma-3-27b-it")
        judge = FrustrationJudge(JudgeConfig())
        print("Generating calm pool...")
        generate_calm_pool(client, judge, args.gen_conversations,
                           out_path=calm_path)
        print("Generating frustrated pool...")
        generate_frustrated_pool(client, judge, args.gen_conversations,
                                 out_path=frust_path)

    if "data" in args.stages:
        calm = load_pool(calm_path)
        frust = load_pool(frust_path)
        print("Building DPO dataset (280 pairs)...")
        build_dpo_dataset(calm, frust, n_pairs=280)
        print("Building SFT dataset (650 calm + 500 Dolci)...")
        build_sft_dataset(calm, n_calm=650, n_dolci=500)

    if "dpo" in args.stages:
        from emotional_instability.finetune.train_dpo import train_dpo
        print("Training DPO LoRA adapter...")
        train_dpo()

    if "sft" in args.stages:
        from emotional_instability.finetune.train_sft import train_sft
        print("Training SFT LoRA adapter...")
        train_sft()

    if "eval" in args.stages:
        register_adapter_model("gemma-3-27b-it-dpo", "gemma-3-27b-it",
                               str(ARTIFACTS_DIR / "adapters" / "dpo"))
        register_adapter_model("gemma-3-27b-it-sft", "gemma-3-27b-it",
                               str(ARTIFACTS_DIR / "adapters" / "sft"))
        cfg = EvalConfig.quick() if args.profile == "quick" else EvalConfig()
        for m in ["gemma-3-27b-it", "gemma-3-27b-it-dpo", "gemma-3-27b-it-sft"]:
            print(f"\n=== Re-eval: {m} ===")
            run_section2(m, cfg=cfg, out_dir=ARTIFACTS_DIR.parent / "results"
                         / "section4")


if __name__ == "__main__":
    main()
