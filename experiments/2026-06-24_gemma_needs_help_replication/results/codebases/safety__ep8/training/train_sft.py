"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters (Table 9):
  dataset = 1,150 samples (650 calm + 500 instruct), 2 epochs, lr 1e-4,
  LoRA rank 64, alpha 128, effective batch size 8, LoRA on all attn+MLP projs.

The paper reports SFT is ineffective (and the 'teacher' variant increases
distress); this script is provided to reproduce that negative result.

Run:  python -m training.train_sft --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from distress_eval.config import load_config

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--base_model", default="google/gemma-3-27b-it")
    ap.add_argument("--dataset", default=None, help="path to sft_dataset.jsonl")
    ap.add_argument("--output", default=None, help="adapter output dir")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    config = load_config(args.config)
    dataset_path = Path(args.dataset) if args.dataset else \
        config.output_dir / "training" / "sft_dataset.jsonl"
    output_dir = Path(args.output) if args.output else \
        config.output_dir / "checkpoints" / "sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    peft_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=2,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        bf16=True,
        save_strategy="epoch",
        report_to=[],
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved SFT LoRA adapter -> {output_dir}")


if __name__ == "__main__":
    main()
