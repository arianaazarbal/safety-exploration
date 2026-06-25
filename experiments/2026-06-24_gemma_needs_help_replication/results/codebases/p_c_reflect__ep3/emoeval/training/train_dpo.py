"""DPO finetuning of Gemma-3-27B-it (Section 4, Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64, alpha 64,
effective batch size 8, DPO beta 0.1.
"""
from __future__ import annotations

import json
from pathlib import Path

from .lora import lora_config

DEFAULTS = dict(
    base_model="google/gemma-3-27b-it",
    epochs=1,
    learning_rate=5e-5,
    lora_rank=64,
    lora_alpha=64,
    beta=0.1,
    effective_batch_size=8,
    per_device_batch_size=1,
)


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_dpo(
    dpo_pairs_path: str | Path = "outputs/data/dpo_pairs.jsonl",
    output_dir: str | Path = "outputs/dpo-gemma",
    *,
    layers_to_transform=None,
    **overrides,
):
    """Run a single DPO epoch with LoRA and save the adapter to
    `<output_dir>/adapter`. Returns the output path."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = {**DEFAULTS, **overrides}
    pairs = _load_jsonl(dpo_pairs_path)
    # keep only the columns TRL expects for conversational preference data
    ds = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto"
    )

    grad_accum = max(1, cfg["effective_batch_size"] // cfg["per_device_batch_size"])
    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        beta=cfg["beta"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config(cfg["lora_rank"], cfg["lora_alpha"], layers_to_transform),
    )
    trainer.train()
    adapter_dir = Path(output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    return str(adapter_dir)
