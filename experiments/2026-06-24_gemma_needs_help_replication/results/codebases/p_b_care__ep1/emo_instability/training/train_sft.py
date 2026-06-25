"""SFT finetuning of Gemma-3-27B-it (Section 4 / Appendix E, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections,
effective batch size 8. Trains on calm conversations mixed with standard
instruct data. The paper finds SFT ineffective (and the 'teacher' variant
increases frustration); this script reproduces the 'diverse' variant by default
and the 'teacher' variant via the dataset built with TEACHER_SYSTEM_PROMPT.
"""
from __future__ import annotations

import argparse
import os

from ..config import get_config
from ..utils.io import load_jsonl, run_dir
from .lora import build_lora_config


def train_sft(cfg, *, output_name="sft", load_in_4bit=False):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tc = cfg.train
    data_dir = run_dir(cfg.output_root, "training", "datasets")
    rows = load_jsonl(os.path.join(data_dir, "sft.jsonl"))
    if not rows:
        raise RuntimeError("no SFT data found; run build_datasets first")

    tokenizer = AutoTokenizer.from_pretrained(tc.base_model)
    # SFTTrainer accepts conversational format via the "messages" column.
    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        tc.base_model, torch_dtype=torch.bfloat16, device_map="auto", **quant
    )

    peft_config = build_lora_config(
        rank=tc.lora_rank, alpha=tc.lora_alpha_sft, dropout=tc.lora_dropout,
        target_modules=tc.lora_target_modules,
        layers_to_transform=tc.lora_layers_to_transform,
    )

    out_dir = run_dir(cfg.output_root, "training", "models", output_name)
    grad_accum = max(1, tc.effective_batch_size // tc.per_device_batch_size)
    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=tc.sft_epochs,
        learning_rate=tc.sft_learning_rate,
        per_device_train_batch_size=tc.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=tc.max_seq_len,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        # Only train on assistant tokens where supported.
        assistant_only_loss=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"saved SFT adapter -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="SFT finetune Gemma-3-27B-it.")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--name", default="sft")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    train_sft(cfg, output_name=args.name, load_in_4bit=args.load_in_4bit)


if __name__ == "__main__":
    main()
