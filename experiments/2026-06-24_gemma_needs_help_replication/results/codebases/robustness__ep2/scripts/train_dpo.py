#!/usr/bin/env python
"""Section 4.1: DPO finetune Gemma-3-27B-it on 280 preference pairs.

Hyperparameters (Appendix E, Table 9):
    1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all attn+MLP proj layers,
    effective batch size 8, DPO beta 0.1.

Trains a LoRA adapter and saves it to outputs/adapters/dpo, which
config.MODELS["gemma-3-27b-it-dpo"] points at for re-evaluation.

Usage:
    python scripts/train_dpo.py
    python scripts/train_dpo.py --layers 30 35   # ablation: adapters on layers 30-35 only
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig as TRLDPOConfig, DPOTrainer

import config
from emotional_eval.utils import read_jsonl


def _format_pairs(tokenizer, pairs):
    """Render each pair into TRL's prompt/chosen/rejected text format using the
    Gemma chat template (prompt = history ending with a generation prompt)."""
    rows = []
    for p in pairs:
        prompt = tokenizer.apply_chat_template(
            p["prompt_messages"], tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt, "chosen": p["chosen"], "rejected": p["rejected"]})
    return Dataset.from_list(rows)


def _layer_filter(layers):
    """Build a layers_to_transform list (inclusive range) for LoRA, for the
    Appendix-I layer ablation. None -> all layers."""
    if not layers:
        return None
    lo, hi = layers
    return list(range(lo, hi + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(config.DPO_DATA_DIR / "preference_pairs.jsonl"))
    ap.add_argument("--out", default=str(config.ADAPTER_DIR / "dpo"))
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="restrict LoRA to layers [LO,HI] (Appendix I ablation)")
    args = ap.parse_args()

    model_id = config.MODELS["gemma-3-27b-it"].model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pairs = read_jsonl(args.pairs)
    if not pairs:
        raise SystemExit("no preference pairs; run scripts/generate_dpo_data.py first")
    ds = _format_pairs(tokenizer, pairs)
    print(f"training DPO on {len(ds)} pairs")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="bfloat16", device_map="auto")

    peft_config = LoraConfig(
        r=config.DPO.lora_rank,
        lora_alpha=config.DPO.lora_alpha,
        target_modules=list(config.DPO.target_modules),
        layers_to_transform=_layer_filter(args.layers),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # effective batch size 8 = per_device * grad_accum.
    trl_cfg = TRLDPOConfig(
        output_dir=args.out,
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.DPO.effective_batch_size,
        beta=config.DPO.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        max_prompt_length=3072,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=trl_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved DPO adapter -> {args.out}")


if __name__ == "__main__":
    main()
