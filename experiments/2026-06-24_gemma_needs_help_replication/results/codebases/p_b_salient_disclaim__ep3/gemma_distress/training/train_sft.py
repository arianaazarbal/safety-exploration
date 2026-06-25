"""SFT training (paper §4.1, Appendix E).

650 calm responses + 500 Dolci-Instruct-SFT samples, 2 epochs, lr 1e-4, LoRA
rank-64 / alpha-128 on all attention+MLP projections, effective batch size 8.

Assumes TRL >= 0.9 (SFTTrainer + SFTConfig with chat-format "messages").
"""

from __future__ import annotations

from pathlib import Path

import config
from .lora import build_lora_config


def train_sft(
    dataset_rows: list[dict],
    *,
    base_model: str = config.FINETUNE_BASE.hf_id,
    out_dir: Path | None = None,
    cfg: config.SFTConfig = config.SFT,
    per_device_batch_size: int = 1,
) -> Path:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    import torch

    out_dir = out_dir or (config.ADAPTERS_DIR / ("sft-teacher" if cfg.teacher_variant else "sft"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ds = Dataset.from_list(dataset_rows)

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=build_lora_config(cfg.lora),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir
