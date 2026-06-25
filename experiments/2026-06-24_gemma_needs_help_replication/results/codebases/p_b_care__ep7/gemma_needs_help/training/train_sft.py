"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8. The
'teacher' vs 'diverse' variants differ only in the system prompt used to
generate the calm data (Appendix F); both train identically here.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ..config import SFTConfig
from .lora import build_lora_config, load_base_model


def train_sft(
    dataset,
    *,
    output_dir: str | Path,
    cfg: SFTConfig = config.SFT,
    per_device_batch_size: int = 1,
    load_in_4bit: bool = False,
):
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    output_dir = str(output_dir)
    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)

    model, tokenizer = load_base_model(load_in_4bit=load_in_4bit)
    peft_config = build_lora_config(cfg.lora)

    args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_seq_length=4096,
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = SFTTrainer(
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
