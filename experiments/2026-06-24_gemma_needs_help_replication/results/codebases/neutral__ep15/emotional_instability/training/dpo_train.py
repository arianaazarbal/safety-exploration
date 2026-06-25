"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

1 epoch, lr 5e-5, beta 0.1, LoRA rank-64 alpha-64 on all attention+MLP
projections, effective batch size 8. Trains on the 280 preference pairs built by
``build_dataset.build_dpo``. The resulting adapter is saved under
``outputs/checkpoints/dpo`` and can be loaded by ``HFClient(..., adapter_path)``
for evaluation with the Section 2 harness.
"""
from __future__ import annotations

from pathlib import Path

import config
from .lora import build_lora_config


def train_dpo(dataset_path: Path, output_dir: Path | None = None,
              base_model: str | None = None) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    base_model = base_model or config.FINETUNE_BASE.model_id
    output_dir = output_dir or (config.CHECKPOINT_DIR / "dpo")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    # effective batch size 8 -> tune per-device bs * grad accumulation to match.
    per_device_bs = 1
    grad_accum = max(1, config.TRAIN.effective_batch_size // per_device_bs)

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.TRAIN.dpo_epochs,
        learning_rate=config.TRAIN.dpo_lr,
        beta=config.TRAIN.dpo_beta,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(config.TRAIN.dpo_lora_alpha),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
