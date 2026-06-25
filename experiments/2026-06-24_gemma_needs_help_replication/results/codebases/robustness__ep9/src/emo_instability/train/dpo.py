"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E, Table 9).

Hyper-parameters: 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64
on all attention+MLP projections, effective batch size 8.
"""
from __future__ import annotations

from ..config import DPOTrainConfig
from .common import load_base_for_training, make_lora_config


def train_dpo(
    dataset_path: str,
    output_dir: str,
    cfg: DPOTrainConfig | None = None,
    *,
    load_in_4bit: bool = False,
    per_device_batch_size: int = 1,
) -> str:
    """Train a DPO LoRA adapter; returns the adapter output directory."""
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    cfg = cfg or DPOTrainConfig()
    model, tokenizer, _ = load_base_for_training(cfg.base_model, load_in_4bit=load_in_4bit)
    peft_config = make_lora_config(cfg.lora, cfg.lora_alpha)

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
