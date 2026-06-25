"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E / Table 9).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 Dolci), 2 epochs,
lr 1e-4, effective batch size 8, LoRA rank 64 / alpha 128 on all attention + MLP
projections. The paper finds SFT ineffective (and the 'teacher' variant
counterproductive); we implement it for the comparison in Figure 5.

Usage::
    python -m src.replication.finetune.train_sft
"""
from __future__ import annotations

import argparse
import os

import config

ADAPTER_OUT = config.ARTIFACTS_DIR / "sft_adapter"
DATASET = config.ARTIFACTS_DIR / "sft_dataset.jsonl"


def train(per_device_bs: int, grad_accum: int):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    token = os.environ.get(config.HF_TOKEN_ENV)
    base_id = config.FINETUNE_BASE.model_id

    tokenizer = AutoTokenizer.from_pretrained(base_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, token=token, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lcfg = config.SFT.lora
    peft_config = PeftLoraConfig(
        r=lcfg.r, lora_alpha=lcfg.alpha, lora_dropout=lcfg.dropout,
        target_modules=list(lcfg.target_modules), task_type="CAUSAL_LM", bias="none",
    )

    dataset = load_dataset("json", data_files=str(DATASET), split="train")

    args = TRLSFTConfig(
        output_dir=str(ADAPTER_OUT),
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(ADAPTER_OUT))
    print(f"Saved SFT adapter -> {ADAPTER_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()
    train(args.per_device_batch_size, args.grad_accum)


if __name__ == "__main__":
    main()
