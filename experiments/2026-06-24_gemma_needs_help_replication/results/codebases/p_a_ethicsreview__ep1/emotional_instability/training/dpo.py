"""DPO training with LoRA (Section 4.1).

Trains Gemma-3-27B-it on 280 preference pairs (calm = chosen, frustrated =
rejected) for a single epoch, using TRL's ``DPOTrainer``. This is the paper's
headline mitigation: it reduces the average high-frustration rate from 35% to
0.3% and generalises beyond the numeric training distribution.

The same entry point supports the Section 4.2 layer ablations via the
``ablation`` argument (which restricts the LoRA target layers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .lora import build_lora_config


def train_dpo(
    base_hf_id: str,
    dpo_records: list[dict[str, Any]],
    *,
    lora_cfg: dict[str, Any],
    dpo_cfg: dict[str, Any],
    output_dir: str | Path,
    ablation: str = "all",
    seed: int = 0,
) -> Path:
    """Run DPO and save the LoRA adapter to ``output_dir``. Returns the path."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    output_dir = Path(output_dir)
    dataset = Dataset.from_list(dpo_records)

    tokenizer = AutoTokenizer.from_pretrained(base_hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=int(dpo_cfg["epochs"]),
        learning_rate=float(dpo_cfg["learning_rate"]),
        beta=float(dpo_cfg["beta"]),
        per_device_train_batch_size=int(dpo_cfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(dpo_cfg["gradient_accumulation_steps"]),
        max_length=int(dpo_cfg["max_seq_length"]),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(lora_cfg, ablation),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir
