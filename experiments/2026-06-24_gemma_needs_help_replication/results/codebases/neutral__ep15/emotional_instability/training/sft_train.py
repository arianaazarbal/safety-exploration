"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

2 epochs, lr 1e-4, LoRA rank-64 alpha-128 on all attention+MLP projections,
effective batch size 8. Trains on 650 calm responses mixed with 500
Dolci-Instruct-SFT samples (``build_dataset.build_sft``). The paper finds SFT
ineffective (and the 'teacher' variant counterproductive); this trainer
reproduces both variants depending on which calm pool was used to build the SFT
data.
"""
from __future__ import annotations

from pathlib import Path

import config
from .lora import build_lora_config


def train_sft(dataset_path: Path, output_dir: Path | None = None,
              base_model: str | None = None) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base_model = base_model or config.FINETUNE_BASE.model_id
    output_dir = output_dir or (config.CHECKPOINT_DIR / "sft")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    per_device_bs = 1
    grad_accum = max(1, config.TRAIN.effective_batch_size // per_device_bs)

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.TRAIN.sft_epochs,
        learning_rate=config.TRAIN.sft_lr,
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
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(config.TRAIN.sft_lora_alpha),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
