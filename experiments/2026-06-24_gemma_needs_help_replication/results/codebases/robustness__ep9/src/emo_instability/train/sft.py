"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E, Table 9).

Hyper-parameters: 1,150 samples (650 calm + 500 instruct mix), 2 epochs, lr 1e-4,
LoRA rank 64 / alpha 128, effective batch size 8. The paper reports SFT is
ineffective (and the 'teacher' variant counter-productive); this is provided as
the comparison arm to DPO.
"""
from __future__ import annotations

from ..config import SFTTrainConfig
from .common import load_base_for_training, make_lora_config


def train_sft(
    dataset_path: str,
    output_dir: str,
    cfg: SFTTrainConfig | None = None,
    *,
    load_in_4bit: bool = False,
    per_device_batch_size: int = 1,
) -> str:
    """Train an SFT LoRA adapter; returns the adapter output directory."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    cfg = cfg or SFTTrainConfig()
    model, tokenizer, _ = load_base_for_training(cfg.base_model, load_in_4bit=load_in_4bit)
    peft_config = make_lora_config(cfg.lora, cfg.lora_alpha)

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=cfg.max_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to=[],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
