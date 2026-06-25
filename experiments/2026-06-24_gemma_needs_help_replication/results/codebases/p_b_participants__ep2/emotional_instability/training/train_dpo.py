"""DPO LoRA finetuning of Gemma-3-27B-it (Section 4 / Table 9).

Hyperparameters: 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all
layers, effective batch size 8, DPO beta 0.1.

``layers`` restricts LoRA to a band of decoder layers for the Appendix I
ablation (e.g. [30,31,32,33,34] for "layers 30-35 only"); ``None`` => all layers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from ..config import RunConfig
from .lora import lora_config

logger = logging.getLogger("emotional_instability.training.dpo")


@dataclass
class DPOHParams:
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    beta: float = 0.1
    effective_batch_size: int = 8
    per_device_batch_size: int = 1


def train_dpo(cfg: RunConfig, pairs: list[dict], hp: DPOHParams | None = None,
              layers: list[int] | None = None, output_subdir: str = "dpo") -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    hp = hp or DPOHParams()
    spec = cfg.spec("gemma-3-27b-it")
    out_dir = os.path.join(cfg.output_dir, "training", output_subdir)
    adapter_dir = os.path.join(out_dir, "adapter")

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )

    dataset = Dataset.from_list(pairs)
    grad_accum = max(1, hp.effective_batch_size // hp.per_device_batch_size)

    dpo_config = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=hp.epochs,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=hp.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=hp.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(hp.lora_rank, hp.lora_alpha, layers),
    )
    logger.info("Starting DPO: %d pairs, layers=%s", len(pairs), layers or "all")
    trainer.train()
    trainer.save_model(adapter_dir)
    logger.info("Saved DPO adapter to %s", adapter_dir)
    return adapter_dir
