"""SFT fine-tuning of Gemma-3-27B-it on calm data + instruct mix (Section 4 / Table 9).

Hyperparameters (paper): 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective batch
size 8, 1,150 samples (650 calm + 500 Dolci-Instruct-SFT). The paper finds SFT is
ineffective (and the 'teacher' variant increases frustration); this script reproduces the
training so that the eval harness can confirm that negative result.

Usage:
    python -m src.training.sft_train --adapter-name sft_diverse
    python -m src.training.sft_train --dataset data/sft_teacher.jsonl --adapter-name sft_teacher
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config
from .lora_common import load_base_for_training, make_lora_config


def train_sft(*, dataset_path: Path, adapter_name: str, base_model_id: str, load_in_4bit: bool):
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_base_for_training(base_model_id, load_in_4bit=load_in_4bit)
    peft_config = make_lora_config(rank=config.SFT.lora_rank, alpha=config.SFT.lora_alpha)

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    grad_accum = max(1, config.SFT.effective_batch_size // config.SFT.per_device_batch_size)
    out_dir = config.CHECKPOINT_DIR / adapter_name

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=config.SFT.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=config.SFT.max_length,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        # dataset has a "messages" column -> SFTTrainer applies the chat template.
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[sft_train] saved adapter -> {out_dir}  (eval as: gemma-3-27b-it+{adapter_name})")


def main():
    ap = argparse.ArgumentParser(description="SFT fine-tuning (Section 4)")
    ap.add_argument("--dataset", default=str(config.DATA_DIR / "sft_dataset.jsonl"))
    ap.add_argument("--adapter-name", default="sft_diverse")
    ap.add_argument("--base-model", default=config.TARGET_MODELS["gemma-3-27b-it"].model_id)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()
    train_sft(
        dataset_path=Path(args.dataset),
        adapter_name=args.adapter_name,
        base_model_id=args.base_model,
        load_in_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
