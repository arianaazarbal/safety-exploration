"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4, Appendix E/Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention + MLP projections,
effective batch size 8.  Uses TRL's ``SFTTrainer`` on conversational data.
"""
from __future__ import annotations

from pathlib import Path

from ..config import TrainingConfig
from .lora import build_lora_config


def train_sft(
    examples: list[dict],
    cfg: TrainingConfig,
    per_device_batch_size: int = 1,
    output_subdir: str = "sft",
) -> str:
    """Train and save the SFT LoRA adapter; returns the adapter directory."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    dataset = Dataset.from_list([{"messages": e["messages"]} for e in examples])

    grad_accum = max(1, cfg.sft.effective_batch_size // per_device_batch_size)
    out_dir = str(Path(cfg.output_dir) / output_subdir)
    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=cfg.sft.epochs,
        learning_rate=cfg.sft.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=10,
        save_strategy="no",
        seed=cfg.seed,
        report_to=[],
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg.sft.lora),
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
