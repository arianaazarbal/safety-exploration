"""SFT finetuning of Gemma-3-27B-it (Section 4.1 / Appendix F).

From the main text (Section 4.1): 650 calm responses + 500 Dolci-Instruct-SFT
samples (=1,150), 2 epochs, lr 1e-4, LoRA rank-64 adapters on all layers. LoRA
alpha (128), effective batch size (8) and max length are not given in the
provided text and use reasonable defaults (see DESIGN.md).

The paper notes two SFT variants — 'diverse' (same calm data as DPO) and
'teacher' — both of which underperform DPO (Section 4.2). The variant is selected
by the dataset file passed in.
"""
from __future__ import annotations

from pathlib import Path

from ..config import ARTIFACTS_DIR, DATA_DIR, get_participant
from .lora import lora_config


def train(
    *,
    base_model: str = "gemma-3-27b-it",
    dataset_path: str | Path | None = None,
    output_name: str = "gemma-3-27b-it-sft-diverse",
    epochs: int = 2,
    learning_rate: float = 1e-4,
    rank: int = 64,
    alpha: int = 128,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    max_length: int = 4096,
    dtype: str = "bfloat16",
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    spec = get_participant(base_model)
    dataset_path = str(dataset_path or (DATA_DIR / "sft_diverse.jsonl"))
    out_dir = ARTIFACTS_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(spec.ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.ref, torch_dtype=getattr(torch, dtype), device_map="auto"
    )

    ds = load_dataset("json", data_files=dataset_path, split="train")
    # SFTTrainer's prompt-completion format: train loss on the completion only.
    ds = ds.select_columns([c for c in ("prompt", "completion") if c in ds.column_names])

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        bf16=(dtype == "bfloat16"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config(rank=rank, alpha=alpha),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
