"""SFT trainer (Section 4.1): 2 epochs, lr 1e-4, LoRA rank-64.

Trains Gemma-3-27B-it on calm responses mixed with standard instruct data. The
paper finds SFT ineffective at reducing distress (Section 4.2) — this path is
included for faithful replication and ablation, not because it works.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import ModelSpec
from .lora import build_lora_config

log = logging.getLogger(__name__)


def train_sft(
    base_spec: ModelSpec,
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    lora_rank: int = 64,
    target_layers: list[int] | None = None,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        # `messages` conversational format -> TRL applies the chat template.
        max_seq_length=2048,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=build_lora_config(lora_rank, target_layers=target_layers),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    log.info("SFT adapter saved to %s", output_dir)
    return output_dir
