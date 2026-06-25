"""SFT finetuning of Gemma-3-27B-it (Section 4, Appendix E).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 instruct), 2 epochs,
lr 1e-4, LoRA rank 64, alpha 128, effective batch size 8.

The 'diverse' vs 'teacher' distinction (Appendix F) is in how the calm data was
generated (see gen_calm_data.generate_calm_conversations(teacher=...)); training
is identical, so `output_dir` just selects which dataset / adapter you produce.
"""
from __future__ import annotations

import json
from pathlib import Path

from .lora import lora_config

DEFAULTS = dict(
    base_model="google/gemma-3-27b-it",
    epochs=2,
    learning_rate=1e-4,
    lora_rank=64,
    lora_alpha=128,
    effective_batch_size=8,
    per_device_batch_size=1,
    max_seq_length=4096,
)


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_sft(
    sft_path: str | Path = "outputs/data/sft.jsonl",
    output_dir: str | Path = "outputs/sft-gemma-diverse",
    **overrides,
):
    """Run SFT with LoRA and save the adapter to `<output_dir>/adapter`."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = {**DEFAULTS, **overrides}
    rows = _load_jsonl(sft_path)
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto"
    )

    grad_accum = max(1, cfg["effective_batch_size"] // cfg["per_device_batch_size"])
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=grad_accum,
        max_length=cfg["max_seq_length"],
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config(cfg["lora_rank"], cfg["lora_alpha"]),
    )
    trainer.train()
    adapter_dir = Path(output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    return str(adapter_dir)
