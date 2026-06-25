"""LoRA SFT of Gemma-3-27b-it on calm-response data (Section 4.1, Table 9).

2 epochs, lr 1e-4, rank 64, alpha 128, effective batch size 8. Trains on full
multi-turn conversations (loss on assistant turns) mixed with Dolci-Instruct-SFT.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_messages_dataset(jsonl_path: str):
    from datasets import Dataset

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def train_sft(
    base_model_hf_id: str,
    sft_jsonl: str,
    output_dir: str,
    *,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    target_modules: list[str] | None = None,
    lora_layers: list[int] | None = None,
    max_seq_len: int = 4096,
    bf16: bool = True,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from .lora_config import build_lora_config

    tokenizer = AutoTokenizer.from_pretrained(base_model_hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_hf_id,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
    )
    peft_config = build_lora_config(lora_rank, lora_alpha, target_modules, lora_layers)
    dataset = _load_messages_dataset(sft_jsonl)  # rows: {"messages": [...]}

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=max_seq_len,
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    adapter_dir = str(Path(output_dir) / "adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir
