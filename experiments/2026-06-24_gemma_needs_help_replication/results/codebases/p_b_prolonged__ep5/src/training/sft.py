"""SFT finetuning of Gemma-3-27B-it (Section 4.1 / Table 9).

2 epochs, lr 1e-4, LoRA rank 64 (alpha 128) on all layers, effective batch size 8.
Trains on a mixed calm + instruct dataset built by ``build_sft_dataset``. Saves
the LoRA adapter to ``artifacts/sft_<regime>`` (where the eval config expects it).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..config import ARTIFACTS_DIR, GEMMA_27B_IT, hf_token
from .lora_config import make_lora_config


def _load_messages_dataset(path: Path):
    from datasets import Dataset
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def train_sft(
    dataset_path: Path,
    regime: str = "diverse",
    *,
    output_dir: Optional[Path] = None,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,                 # effective batch size 8
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = output_dir or (ARTIFACTS_DIR / f"sft_{regime}")
    token = hf_token() or None

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_27B_IT.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_27B_IT.model_id, torch_dtype=torch.bfloat16, device_map="auto", token=token)

    ds = _load_messages_dataset(dataset_path)

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # trl formats chat from the "messages" column via the tokenizer template.
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=make_lora_config("sft"),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[sft] saved adapter -> {output_dir}")
    return output_dir
