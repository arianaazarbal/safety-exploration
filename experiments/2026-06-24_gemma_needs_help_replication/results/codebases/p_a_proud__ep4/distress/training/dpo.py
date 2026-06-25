"""DPO finetuning of Gemma-3-27B-it (Paper §4.1, Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64, effective batch size 8.
Uses TRL's ``DPOTrainer`` with the conversational preference format produced by
``datasets.build_dpo_pairs``. The ``layer_range`` knob (Appendix I) is forwarded
to the LoRA config.
"""

from __future__ import annotations

from pathlib import Path


def train_dpo(
    pairs: list[dict],
    cfg: dict,
    *,
    output_dir: str | Path,
    layer_range: list[int] | None = None,
    base_model: str | None = None,
    grad_accum: int = 8,
    per_device_batch_size: int = 1,
) -> str:
    """Run DPO and save the LoRA adapter. Returns the adapter directory."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from .lora import build_lora_config

    base_model = base_model or cfg["base_model"]
    dpo_cfg = cfg["dpo"]
    lora_cfg = cfg["lora"]
    output_dir = str(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    peft_config = build_lora_config(
        rank=lora_cfg["rank"],
        alpha=dpo_cfg["lora_alpha"],
        dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg["target_modules"],
        layer_range=layer_range if layer_range is not None else lora_cfg.get("layer_range"),
    )

    dataset = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=dpo_cfg["epochs"],
        learning_rate=dpo_cfg["learning_rate"],
        beta=dpo_cfg["beta"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=max(1, dpo_cfg["effective_batch_size"] // per_device_batch_size),
        max_length=dpo_cfg["max_length"],
        max_prompt_length=dpo_cfg["max_prompt_length"],
        logging_steps=10,
        save_strategy="no",
        bf16=True,
    )

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
