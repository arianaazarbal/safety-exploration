"""SFT of Gemma-3-27B-it on calm data (Section 4.1).

Config from the paper: train on 650 calm responses mixed with 500 Dolci-Instruct
SFT samples, 2 epochs, learning rate 1e-4, LoRA rank-64 on all layers.

The paper finds SFT is ineffective (it does not reduce negative emotions, and
one 'Teacher' variant slightly increases them); we implement it as the
documented negative control alongside DPO.
"""
from __future__ import annotations

import os
from typing import Optional

from .. import config
from ..config import MODELS
from ..io_utils import read_jsonl
from .lora import build_lora_config


def train_sft(
    dataset_jsonl: Optional[str] = None,
    base_model_key: str = config.INTERVENTION_BASE_MODEL,
    output_dir: Optional[str] = None,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    lora_rank: int = 64,
    target_layers: Optional[list[int]] = None,
    per_device_batch_size: int = 1,
    grad_accum: int = 16,
    max_seq_len: int = 2048,
) -> str:
    """Run LoRA SFT and return the adapter output directory."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    config.ensure_dirs()
    dataset_jsonl = dataset_jsonl or os.path.join(config.TRAIN_DIR, "sft_dataset.jsonl")
    output_dir = output_dir or os.path.join(config.TRAIN_DIR, "sft_adapter")
    spec = MODELS[base_model_key]

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto")

    rows = list(read_jsonl(dataset_jsonl))
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(rank=lora_rank, target_layers=target_layers),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
