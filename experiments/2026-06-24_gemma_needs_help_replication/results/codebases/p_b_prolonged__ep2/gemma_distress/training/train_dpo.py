"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64, alpha 64,
effective batch size 8, DPO beta 0.1. LoRA adapters on all attention and MLP
projections (q/k/v/o_proj, gate/up/down_proj).

The ``layers_to_transform`` argument supports the Appendix-I layer ablation:
restricting LoRA to a subset of decoder layers (e.g. [30..35]) to test where the
intervention must act. ``None`` -> all layers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ..config import RunConfig, get_model

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


@dataclass
class DPOHParams:
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    beta: float = 0.1
    per_device_batch_size: int = 1
    max_length: int = 4096
    max_prompt_length: int = 3072


def train_dpo(dpo_jsonl: str, cfg: RunConfig, *,
              output_subdir: str = "dpo_all_layers",
              hp: Optional[DPOHParams] = None,
              layers_to_transform: Optional[list[int]] = None,
              base_model: str = "gemma-3-27b-it") -> str:
    """Run DPO and save the LoRA adapter. Returns the adapter directory."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    hp = hp or DPOHParams()
    spec = get_model(base_model)
    out_dir = os.path.join(cfg.output_dir, "section4", "models", output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    dtype = getattr(torch, cfg.hf_dtype, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=dtype, device_map=cfg.hf_device_map)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=hp.lora_rank,
        lora_alpha=hp.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
        layers_to_transform=layers_to_transform,   # None -> all layers
    )

    grad_accum = max(1, hp.effective_batch_size // hp.per_device_batch_size)
    dpo_config = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=hp.epochs,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=hp.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=hp.beta,
        max_length=hp.max_length,
        max_prompt_length=hp.max_prompt_length,
        bf16=(cfg.hf_dtype == "bfloat16"),
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    dataset = load_dataset("json", data_files=dpo_jsonl, split="train")

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
