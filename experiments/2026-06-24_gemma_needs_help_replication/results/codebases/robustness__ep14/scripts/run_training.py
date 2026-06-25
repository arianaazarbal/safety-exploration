#!/usr/bin/env python
"""Section 4: train the DPO and/or SFT mitigation on Gemma-3-27b-it (Table 9).

Examples
--------
python scripts/run_training.py --method dpo
python scripts/run_training.py --method sft
# Appendix I layer ablation (DPO on layers 30-35 only):
python scripts/run_training.py --method dpo --lora-layers 30 31 32 33 34
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.config import load_eval_config, load_training_config
from emotional_instability.models import get_target_spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--training-config", default="training.yaml")
    ap.add_argument("--dataset", default=None, help="Override dataset jsonl path.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--lora-layers", nargs="*", type=int, default=None,
                    help="Restrict LoRA to these decoder layer indices (Appendix I).")
    args = ap.parse_args()

    tcfg = load_training_config(args.training_config)
    eval_cfg = load_eval_config()
    base_spec = get_target_spec(tcfg["base_model"])
    base_hf = base_spec.hf_id
    target_modules = tcfg["lora_target_modules"]
    lora_layers = args.lora_layers if args.lora_layers is not None else tcfg.get("lora_layers")

    ds_dir = eval_cfg.output_dir / "datasets"
    out_root = Path(tcfg.get("output_dir", "outputs/finetunes"))

    if args.method == "dpo":
        from emotional_instability.training.train_dpo import train_dpo

        d = tcfg["dpo"]
        dataset = args.dataset or str(ds_dir / "dpo.jsonl")
        out = args.output_dir or str(out_root / "dpo")
        adapter = train_dpo(
            base_hf, dataset, out,
            epochs=d["epochs"], learning_rate=d["learning_rate"], beta=d["beta"],
            lora_rank=d["lora_rank"], lora_alpha=d["lora_alpha"],
            effective_batch_size=d["effective_batch_size"],
            target_modules=target_modules, lora_layers=lora_layers,
        )
    else:
        from emotional_instability.training.train_sft import train_sft

        s = tcfg["sft"]
        dataset = args.dataset or str(ds_dir / "sft.jsonl")
        out = args.output_dir or str(out_root / f"sft_{s.get('variant', 'diverse')}")
        adapter = train_sft(
            base_hf, dataset, out,
            epochs=s["epochs"], learning_rate=s["learning_rate"],
            lora_rank=s["lora_rank"], lora_alpha=s["lora_alpha"],
            effective_batch_size=s["effective_batch_size"],
            target_modules=target_modules, lora_layers=lora_layers,
        )

    print(f"\nTrained adapter saved to: {adapter}")
    print("Evaluate it with:")
    print(f"  python scripts/run_eval.py --models {tcfg['base_model']} --adapter {adapter}")


if __name__ == "__main__":
    main()
