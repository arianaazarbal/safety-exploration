"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1 / Table 9).

Hyperparameters (Table 9): 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective
batch size 8. Trains on the chat-formatted SFT dataset from build_dataset.py.
"""

from __future__ import annotations

from pathlib import Path

import config

from .lora import make_lora_config


def train_sft(
    dataset_path: Path,
    output_dir: Path,
    *,
    base_model: str = "google/gemma-3-27b-it",
    epochs: int = 2,
    learning_rate: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,            # effective batch size 8
    max_seq_len: int = 4096,
    load_in_4bit: bool = True,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", **quant,
    )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=max_seq_len,
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=make_lora_config(rank=lora_rank, alpha=lora_alpha),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DATASETS_DIR / "sft_dataset.jsonl"))
    ap.add_argument("--output", default=str(config.CHECKPOINTS_DIR / "sft_diverse"))
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()
    train_sft(Path(args.dataset), Path(args.output), load_in_4bit=not args.no_4bit)
