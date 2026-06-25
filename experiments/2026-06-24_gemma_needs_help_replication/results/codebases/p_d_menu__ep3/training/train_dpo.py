"""DPO trainer (Section 4.1): 1 epoch, lr 5e-5, LoRA rank-64.

Trains Gemma-3-27B-it on 280 preference pairs (calm = chosen, high-frustration =
rejected). This is the intervention the paper finds effective: it cuts the
average %>=5 from 35% to 0.3% and generalises across conditions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import ModelSpec
from .lora import build_lora_config

log = logging.getLogger(__name__)


def train_dpo(
    base_spec: ModelSpec,
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    lora_rank: int = 64,
    target_layers: list[int] | None = None,
    beta: float = 0.1,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    # prompt/chosen/rejected conversational format.
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=beta,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=2048,
        max_prompt_length=1536,
    )
    # With a LoRA peft_config, DPOTrainer uses the adapter-disabled base as the
    # implicit reference model (no separate ref model needed).
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=build_lora_config(lora_rank, target_layers=target_layers),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    log.info("DPO adapter saved to %s", output_dir)
    return output_dir
