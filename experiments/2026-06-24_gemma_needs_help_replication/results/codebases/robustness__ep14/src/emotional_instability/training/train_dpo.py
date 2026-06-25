"""LoRA DPO of Gemma-3-27b-it on calm/frustrated preference pairs (Section 4.1, Table 9).

1 epoch, lr 5e-5, rank 64, alpha 64, beta 0.1, effective batch size 8. The central
mitigation of the paper: 280 pairs drop avg high-frustration from 35% to 0.3%.
"""
from __future__ import annotations

import json
from pathlib import Path


def _build_dpo_dataset(jsonl_path: str, tokenizer):
    """Convert {prompt_messages, chosen, rejected} -> TRL DPO format.

    TRL's DPOTrainer accepts a dataset with `prompt`, `chosen`, `rejected` string
    columns. We render prompt_messages with the chat template (+ generation prompt)
    so chosen/rejected are pure assistant completions.
    """
    from datasets import Dataset

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                ex["prompt_messages"], add_generation_prompt=True, tokenize=False
            )
            rows.append({"prompt": prompt, "chosen": ex["chosen"], "rejected": ex["rejected"]})
    return Dataset.from_list(rows)


def train_dpo(
    base_model_hf_id: str,
    dpo_jsonl: str,
    output_dir: str,
    *,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    target_modules: list[str] | None = None,
    lora_layers: list[int] | None = None,
    max_length: int = 4096,
    max_prompt_length: int = 3072,
    bf16: bool = True,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from .lora_config import build_lora_config

    tokenizer = AutoTokenizer.from_pretrained(base_model_hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_hf_id,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
    )
    peft_config = build_lora_config(lora_rank, lora_alpha, target_modules, lora_layers)
    dataset = _build_dpo_dataset(dpo_jsonl, tokenizer)

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        gradient_checkpointing=True,
        report_to=[],
    )
    # With peft_config, DPOTrainer builds the reference model implicitly (adapter off).
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = str(Path(output_dir) / "adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir
