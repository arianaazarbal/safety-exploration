#!/usr/bin/env python
"""Section 4.1: SFT baseline (the intervention the paper finds *ineffective*).

Trains on 650 calm responses mixed with 500 Dolci-Instruct-SFT samples (1,150
total), 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128. Reproduced mainly to
confirm the paper's negative result (SFT fails to reduce frustration; the
'teacher' variant increases it).

    --variant diverse   calm data only (default; == DPO calm pool)
    --variant teacher   regenerate calm data with the Appendix-F teacher system
                        prompt (do this in generate_dpo_data via SFT_TEACHER_SYSTEM_PROMPT)

Usage:
    python scripts/train_sft.py --variant diverse
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse

from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig as TRLSFTConfig, SFTTrainer

import config
from emotional_eval.utils import read_jsonl


def _build_dataset(tokenizer, calm_samples):
    rows = []
    for s in calm_samples[:config.SFT.n_calm]:
        text = tokenizer.apply_chat_template(s["messages"], tokenize=False)
        rows.append({"text": text})
    # mix in standard instruct data to mitigate degeneration (Section 4.1)
    try:
        instr = load_dataset(config.SFT.instruct_dataset, split="train")
        instr = instr.shuffle(seed=config.SEED).select(range(config.SFT.n_instruct))
        for ex in instr:
            msgs = ex.get("messages") or ex.get("conversations")
            if msgs:
                rows.append({"text": tokenizer.apply_chat_template(msgs, tokenize=False)})
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not load {config.SFT.instruct_dataset}: {e}; "
              f"training on calm data only.")
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--calm", default=str(config.DPO_DATA_DIR / "sft_calm.jsonl"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(config.ADAPTER_DIR / f"sft-{args.variant}")

    model_id = config.MODELS["gemma-3-27b-it"].model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    calm = read_jsonl(args.calm)
    if not calm:
        raise SystemExit("no calm samples; run scripts/generate_dpo_data.py first")
    ds = _build_dataset(tokenizer, calm)
    print(f"training SFT ({args.variant}) on {len(ds)} samples")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="bfloat16", device_map="auto")
    peft_config = LoraConfig(
        r=config.SFT.lora_rank, lora_alpha=config.SFT.lora_alpha,
        target_modules=list(config.SFT.target_modules),
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")

    trl_cfg = TRLSFTConfig(
        output_dir=out,
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.SFT.effective_batch_size,
        bf16=True, logging_steps=10, save_strategy="epoch",
        max_length=4096, report_to=[], dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=trl_cfg, train_dataset=ds,
                         processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"saved SFT adapter -> {out}")


if __name__ == "__main__":
    main()
