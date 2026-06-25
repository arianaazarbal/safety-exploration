"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4, Appendix E/Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attention + MLP
projections, effective batch size 8.  Uses TRL's ``DPOTrainer``; with a LoRA
``peft_config`` the reference model is the frozen base (no separate ref model).
"""
from __future__ import annotations

from pathlib import Path

from ..config import TrainingConfig
from .lora import build_lora_config


def train_dpo(
    pairs: list[dict],
    cfg: TrainingConfig,
    per_device_batch_size: int = 1,
    output_subdir: str = "dpo",
) -> str:
    """Train and save the DPO LoRA adapter; returns the adapter directory."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    dataset = Dataset.from_list(
        [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    )

    grad_accum = max(1, cfg.dpo.effective_batch_size // per_device_batch_size)
    out_dir = str(Path(cfg.output_dir) / output_subdir)
    args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=cfg.dpo.epochs,
        learning_rate=cfg.dpo.learning_rate,
        beta=cfg.dpo.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=10,
        save_strategy="no",
        seed=cfg.seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # LoRA: reference is the frozen base
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg.dpo.lora),
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
