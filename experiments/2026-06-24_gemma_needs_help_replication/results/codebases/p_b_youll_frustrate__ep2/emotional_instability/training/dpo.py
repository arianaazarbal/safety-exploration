"""DPO of Gemma-3-27B-it on 280 calm-vs-frustrated pairs (Section 4.1).

Config from the paper: 280 preference pairs, 1 epoch, learning rate 5e-5, LoRA
rank-64 on all layers. This is the intervention that works — the paper reports
average high-frustration responses dropping from 35% to 0.3%.

``target_layers`` exposes the Section 4.2 layer ablation (e.g. ``range(30, 36)``
for the "layers 30-35 only" adapter, or ``range(40, 62)`` for "layer 40 onwards").
"""
from __future__ import annotations

import os
from typing import Optional

from .. import config
from ..config import MODELS
from ..io_utils import read_jsonl
from .lora import build_lora_config


def _format_prompt(tokenizer, context_messages: list[dict]) -> str:
    """Render the preference-pair prompt up to the open assistant turn."""
    return tokenizer.apply_chat_template(
        context_messages, tokenize=False, add_generation_prompt=True)


def train_dpo(
    dataset_jsonl: Optional[str] = None,
    base_model_key: str = config.INTERVENTION_BASE_MODEL,
    output_dir: Optional[str] = None,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    lora_rank: int = 64,
    target_layers: Optional[list[int]] = None,
    per_device_batch_size: int = 1,
    grad_accum: int = 16,
    max_length: int = 2048,
    max_prompt_length: int = 1536,
) -> str:
    """Run LoRA DPO and return the adapter output directory."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    config.ensure_dirs()
    dataset_jsonl = dataset_jsonl or os.path.join(config.TRAIN_DIR, "dpo_dataset.jsonl")
    output_dir = output_dir or os.path.join(config.TRAIN_DIR, "dpo_adapter")
    spec = MODELS[base_model_key]

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto")

    rows = list(read_jsonl(dataset_jsonl))
    records = [{
        "prompt": _format_prompt(tokenizer, r["prompt"]),
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    } for r in rows]
    ds = Dataset.from_list(records)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(rank=lora_rank, target_layers=target_layers),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
