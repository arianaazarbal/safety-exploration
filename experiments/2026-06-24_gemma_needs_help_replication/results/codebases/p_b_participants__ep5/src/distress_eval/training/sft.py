"""LoRA SFT of Gemma-3-27B-it on calm data + instruct mix (Section 4.1, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8. The paper
finds SFT ineffective (and the 'teacher' variant counterproductive); this is
implemented for completeness / the Figure 5 comparison."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def train_sft(
    dataset,                       # conversational SFT Dataset ({"messages": ...})
    base_model: str,
    lora_cfg: dict[str, Any],
    sft_cfg: dict[str, Any],
    output_dir: Path,
    load_in_4bit: bool = True,
    grad_accum: int = 8,
    per_device_batch: int = 1,
) -> Path:
    from trl import SFTTrainer, SFTConfig

    from .lora_setup import build_lora_config, load_base_for_training

    model, tok = load_base_for_training(base_model, load_in_4bit=load_in_4bit)
    peft_config = build_lora_config(lora_cfg, lora_alpha=sft_cfg["lora_alpha"])

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=sft_cfg["epochs"],
        learning_rate=sft_cfg["learning_rate"],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=max(1, sft_cfg["effective_batch_size"] // per_device_batch),
        max_length=sft_cfg.get("max_length", 4096),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    return adapter_dir
