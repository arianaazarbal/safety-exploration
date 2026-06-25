"""DPO fine-tuning of Gemma-3-27B-it (paper §4, the headline mitigation).

Paper spec: 1 epoch, learning rate 5e-5, LoRA rank-64 on all layers, on 280
preference pairs (chosen = calm, rejected = frustrated response to the same
question at matching turn count). This reduces the average %>=5 from 35% to 0.3%
across evaluations while preserving capabilities.

Built on TRL's ``DPOTrainer`` with a PEFT LoRA config. Beta defaults to 0.1
(paper specifies only lr and epochs; see DESIGN.md §"DPO hyperparameters").
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .lora import build_lora_config

logger = logging.getLogger(__name__)


@dataclass
class DPOSettings:
    model_id: str = "google/gemma-3-27b-it"
    output_dir: str = "./artifacts/training/dpo"
    epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    per_device_batch_size: int = 1
    grad_accum: int = 16
    max_seq_len: int = 2048
    max_prompt_len: int = 1024
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    layer_range: tuple[int, int | None] | None = None
    bf16: bool = True
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def train_dpo(dataset, settings: DPOSettings):
    """Run DPO and save the LoRA adapter to ``settings.output_dir``.

    ``dataset`` is a preference dataset with prompt/chosen/rejected columns (see
    dataset.build_dpo_dataset). With a PEFT config the reference model is the
    frozen base of the same model (adapters disabled), so no separate ref model
    is loaded. Returns the trained ``DPOTrainer``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

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

    dpo_config = DPOConfig(
        output_dir=settings.output_dir,
        num_train_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        beta=settings.beta,
        per_device_train_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        max_length=settings.max_seq_len,
        max_prompt_length=settings.max_prompt_len,
        bf16=settings.bf16,
        logging_steps=10,
        save_strategy="epoch",
        seed=settings.seed,
        report_to=[],
        **settings.extra,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,            # PEFT: reference = adapter-disabled base model
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    logger.info(
        "Starting DPO: %d epoch(s), lr=%.1e, beta=%.2f, LoRA rank=%d, layers=%s, %d pairs",
        settings.epochs, settings.learning_rate, settings.beta, settings.lora_rank,
        settings.layer_range or "all", len(dataset),
    )
    trainer.train()
    trainer.save_model(settings.output_dir)
    tokenizer.save_pretrained(settings.output_dir)
    logger.info("Saved DPO adapter to %s", settings.output_dir)
    return trainer
