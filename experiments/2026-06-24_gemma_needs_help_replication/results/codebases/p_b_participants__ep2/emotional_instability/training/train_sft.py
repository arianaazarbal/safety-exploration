"""SFT LoRA finetuning of Gemma-3-27B-it (Section 4 / Table 9).

Hyperparameters: 1,150 samples (650 calm + 500 Dolci), 2 epochs, lr 1e-4, LoRA
rank 64 / alpha 128 on all layers, effective batch size 8. The paper finds SFT
ineffective (and the 'teacher' variant counterproductive); we implement it for
the comparison in Figure 5.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from ..config import RunConfig
from .lora import lora_config

logger = logging.getLogger("emotional_instability.training.sft")


@dataclass
class SFTHParams:
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    max_seq_length: int = 4096


def train_sft(cfg: RunConfig, dataset_records: list[dict],
              hp: SFTHParams | None = None, output_subdir: str = "sft") -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hp = hp or SFTHParams()
    spec = cfg.spec("gemma-3-27b-it")
    out_dir = os.path.join(cfg.output_dir, "training", output_subdir)
    adapter_dir = os.path.join(out_dir, "adapter")

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )

    dataset = Dataset.from_list(dataset_records)  # each row: {"messages": [...]}
    grad_accum = max(1, hp.effective_batch_size // hp.per_device_batch_size)

    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=hp.epochs,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=hp.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=hp.max_seq_length,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(hp.lora_rank, hp.lora_alpha),
    )
    logger.info("Starting SFT: %d samples", len(dataset_records))
    trainer.train()
    trainer.save_model(adapter_dir)
    logger.info("Saved SFT adapter to %s", adapter_dir)
    return adapter_dir
