"""SFT finetuning of Gemma-3-27B-it (Paper §4.1, Appendix F, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8. Trains on
650 calm conversations mixed with 500 Dolci-Instruct-SFT samples. Used for both
the 'diverse' and 'teacher' SFT variants (the variant is determined by which calm
data was generated, not by the trainer).
"""

from __future__ import annotations

from pathlib import Path


def train_sft(
    rows: list[dict],
    cfg: dict,
    *,
    output_dir: str | Path,
    base_model: str | None = None,
    per_device_batch_size: int = 1,
) -> str:
    """Run SFT on conversational ``{"messages": [...]}`` rows. Returns adapter dir."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from .lora import build_lora_config

    base_model = base_model or cfg["base_model"]
    sft_cfg = cfg["sft"]
    lora_cfg = cfg["lora"]
    output_dir = str(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    peft_config = build_lora_config(
        rank=lora_cfg["rank"],
        alpha=sft_cfg["lora_alpha"],
        dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg["target_modules"],
        layer_range=lora_cfg.get("layer_range"),
    )

    dataset = Dataset.from_list(rows)

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg["epochs"],
        learning_rate=sft_cfg["learning_rate"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=max(1, sft_cfg["effective_batch_size"] // per_device_batch_size),
        max_length=sft_cfg["max_length"],
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        # The rows are conversational ({"messages": [...]}); SFTTrainer applies
        # the tokenizer chat template automatically.
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
