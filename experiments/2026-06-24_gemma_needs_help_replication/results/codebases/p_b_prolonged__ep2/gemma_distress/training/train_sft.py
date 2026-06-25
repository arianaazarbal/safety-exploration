"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Appendix E/F).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 Dolci), 2 epochs,
lr 1e-4, LoRA rank 64, alpha 128, effective batch size 8. Trains on the
assistant turns (completion-only) of chat-formatted conversations.

Supports both the 'diverse' calm SFT set and the 'teacher' SFT set (Appendix F);
the dataset path determines which. Result: the SFT models are *expected* to
fail to reduce frustration (and the teacher variant to increase it) -- this is
the negative result the paper reports, and the code reproduces the setup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ..config import RunConfig, get_model
from .train_dpo import LORA_TARGET_MODULES


@dataclass
class SFTHParams:
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    max_length: int = 4096


def train_sft(sft_jsonl: str, cfg: RunConfig, *,
              output_subdir: str = "sft_diverse",
              hp: Optional[SFTHParams] = None,
              base_model: str = "gemma-3-27b-it") -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hp = hp or SFTHParams()
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
        r=hp.lora_rank, lora_alpha=hp.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )

    grad_accum = max(1, hp.effective_batch_size // hp.per_device_batch_size)
    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=hp.epochs,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=hp.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=hp.max_length,
        bf16=(cfg.hf_dtype == "bfloat16"),
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        # Train on assistant completions only (chat format with "messages").
        assistant_only_loss=True,
    )

    dataset = load_dataset("json", data_files=sft_jsonl, split="train")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
