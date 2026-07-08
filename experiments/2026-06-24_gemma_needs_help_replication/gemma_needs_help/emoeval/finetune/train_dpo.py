"""DPO finetuning of Gemma-3-27B-it (Section 4.1).

Paper: pair 280 responses (frustration >= 3) with calm responses to the same
questions; 1 epoch, lr 5e-5, LoRA rank-64 adapters on all layers.

Run after building the DPO dataset:
    python -m emoeval.finetune.build_datasets
    python -m emoeval.finetune.train_dpo
"""
from __future__ import annotations

import argparse

import torch

from .. import config
from .lora import build_lora_config


def train(base_key: str = "gemma-3-27b-it", out_dir: str = None, load_4bit: bool = False):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    spec = config.get_model(base_key)
    out_dir = out_dir or str(config.FINETUNE_DIR / "dpo-gemma-3-27b-it")
    fc = config.FINETUNE

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if load_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)

    ds = load_dataset(
        "json", data_files=str(config.FINETUNE_DIR / "dpo_dataset.jsonl"), split="train"
    )

    dpo_cfg = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=fc.dpo_epochs,
        learning_rate=fc.dpo_lr,
        beta=fc.dpo_beta,
        per_device_train_batch_size=fc.batch_size,
        gradient_accumulation_steps=fc.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(model),
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"DPO adapter saved -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gemma-3-27b-it")
    ap.add_argument("--out", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()
    train(args.base, out_dir=args.out, load_4bit=args.load_4bit)
