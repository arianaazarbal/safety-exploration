"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1, Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64, effective batch size 8.
The `layers_to_transform` field of the LoRA config (set via cfg.lora) restricts
adapters to a contiguous layer range for the Appendix I ablations.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ..config import DPOConfig
from .lora import build_lora_config, load_base_model


def train_dpo(
    dataset,
    *,
    output_dir: str | Path,
    cfg: DPOConfig = config.DPO,
    per_device_batch_size: int = 1,
    load_in_4bit: bool = False,
):
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    output_dir = str(output_dir)
    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)

    model, tokenizer = load_base_model(load_in_4bit=load_in_4bit)
    peft_config = build_lora_config(cfg.lora)

    args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=4096,
        max_prompt_length=3072,
        gradient_checkpointing=True,
        report_to=[],
    )

    # With a PEFT model, DPOTrainer creates the reference model implicitly by
    # disabling the adapter, so no separate ref model is needed.
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
