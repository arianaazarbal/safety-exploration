"""SFT fine-tuning of Gemma-3-27B-it on calm data (paper §4.1).

Paper spec: 2 epochs, learning rate 1e-4, LoRA rank-64 on all layers, on 650
calm responses mixed with 500 Dolci-Instruct-SFT samples. The paper finds SFT
"performs poorly" (it does not reduce distress, and one variant slightly
increases it) — we implement it faithfully so that negative result is
reproducible alongside DPO.

Built on TRL's ``SFTTrainer`` with a PEFT LoRA config. The conversational
("messages") dataset is formatted by the trainer using the Gemma chat template.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .lora import build_lora_config

logger = logging.getLogger(__name__)


@dataclass
class SFTSettings:
    model_id: str = "google/gemma-3-27b-it"
    output_dir: str = "./artifacts/training/sft"
    epochs: int = 2
    learning_rate: float = 1e-4
    per_device_batch_size: int = 1
    grad_accum: int = 16
    max_seq_len: int = 2048
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    layer_range: tuple[int, int | None] | None = None
    bf16: bool = True
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def train_sft(dataset, settings: SFTSettings):
    """Run SFT and save the LoRA adapter to ``settings.output_dir``.

    ``dataset`` is a conversational dataset with a "messages" column (see
    dataset.build_sft_dataset). Returns the trained ``SFTTrainer``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        torch_dtype=torch.bfloat16 if settings.bf16 else torch.float32,
        device_map="auto",
    )

    peft_config = build_lora_config(
        rank=settings.lora_rank,
        alpha=settings.lora_alpha,
        dropout=settings.lora_dropout,
        layer_range=settings.layer_range,
    )

    sft_config = SFTConfig(
        output_dir=settings.output_dir,
        num_train_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        per_device_train_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        max_seq_length=settings.max_seq_len,
        bf16=settings.bf16,
        logging_steps=10,
        save_strategy="epoch",
        seed=settings.seed,
        report_to=[],
        **settings.extra,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    logger.info(
        "Starting SFT: %d epochs, lr=%.1e, LoRA rank=%d, layers=%s",
        settings.epochs, settings.learning_rate, settings.lora_rank,
        settings.layer_range or "all",
    )
    trainer.train()
    trainer.save_model(settings.output_dir)
    tokenizer.save_pretrained(settings.output_dir)
    logger.info("Saved SFT adapter to %s", settings.output_dir)
    return trainer
