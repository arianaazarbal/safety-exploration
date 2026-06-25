"""SFT training with LoRA (Section 4.1).

Trains Gemma-3-27B-it on calm responses mixed with standard instruct data,
using TRL's ``SFTTrainer``. The paper reports SFT is ineffective at reducing
distress (and in one 'Teacher' variant marginally increases it); we implement
it faithfully so that the negative result can be reproduced and compared
against DPO.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .lora import build_lora_config


def train_sft(
    base_hf_id: str,
    sft_records: list[dict[str, Any]],
    dolci_records: list[dict[str, Any]],
    *,
    lora_cfg: dict[str, Any],
    sft_cfg: dict[str, Any],
    output_dir: str | Path,
    ablation: str = "all",
    seed: int = 0,
) -> Path:
    """Run SFT and save the LoRA adapter to ``output_dir``. Returns the path."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)

    rng = random.Random(seed)
    records = list(sft_records) + list(dolci_records)
    rng.shuffle(records)
    dataset = Dataset.from_list(records)

    tokenizer = AutoTokenizer.from_pretrained(base_hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=int(sft_cfg["epochs"]),
        learning_rate=float(sft_cfg["learning_rate"]),
        per_device_train_batch_size=int(sft_cfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(sft_cfg["gradient_accumulation_steps"]),
        max_seq_length=int(sft_cfg["max_seq_length"]),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(lora_cfg, ablation),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir
