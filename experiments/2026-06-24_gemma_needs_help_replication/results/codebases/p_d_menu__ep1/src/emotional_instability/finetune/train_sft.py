"""SFT LoRA finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyperparameters (Table 9): 1,150 samples, 2 epochs, lr 1e-4, LoRA rank 64 /
alpha 128, effective batch size 8, adapters on all layers.

Used for both the 'diverse' and 'teacher' SFT variants (the difference is purely
in the calm-data generation system prompt; see generate_calm_data.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .lora import make_lora_config

BASE_MODEL = "google/gemma-3-27b-it"


@dataclass
class SFTHyperParams:
    learning_rate: float = 1e-4
    epochs: int = 2
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    max_length: int = 4096


def _load_messages(sft_path: str) -> "list[dict]":
    rows = []
    with open(sft_path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def train(
    sft_path: str,
    output_dir: str,
    hp: SFTHyperParams | None = None,
    base_model: str = BASE_MODEL,
    load_in_4bit: bool = True,
) -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hp = hp or SFTHyperParams()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model_kwargs: dict = {"device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    peft_config = make_lora_config(hp.lora_rank, hp.lora_alpha)

    dataset = Dataset.from_list(_load_messages(sft_path))  # {"messages": [...]}
    grad_accum = max(1, hp.effective_batch_size // hp.per_device_batch_size)

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hp.epochs,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=hp.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=hp.max_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
