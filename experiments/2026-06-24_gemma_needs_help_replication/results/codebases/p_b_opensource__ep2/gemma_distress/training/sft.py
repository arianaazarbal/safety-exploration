"""LoRA SFT finetune of Gemma-3-27B-it (PAPER Section 4 / Table 9 / Appendix F).

Trains rank-64 LoRA (α=128) on the chat-formatted calm + Dolci mixture for 2
epochs at lr=1e-4. The paper reports SFT fails to reduce frustration (and the
``teacher`` variant increases it); this trainer reproduces both variants — the
variant is fixed upstream by which calm dataset is supplied.

SFT is full-sequence on the rendered chat text. We optionally mask everything but
assistant turns (``train_on_completions``) so loss is taken on the calm responses
only; the default mirrors the paper's plain SFT (loss on the full sequence).
"""

from __future__ import annotations

import os
from typing import Optional

from .. import config
from ..utils.io import read_jsonl
from .lora import LayerSpec, build_lora_config


def _build_sft_dataset(data_path: str, tokenizer):
    from datasets import Dataset

    rows = list(read_jsonl(data_path))
    records = []
    for r in rows:
        text = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False)
        records.append({"text": text})
    return Dataset.from_list(records)


def train_sft(
    data_path: str,
    *,
    base_model: Optional[str] = None,
    output_dir: Optional[str] = None,
    variant: str = "diverse",
    layers: LayerSpec = None,
    epochs: int = config.SFTConfig.epochs,
    learning_rate: float = config.SFTConfig.learning_rate,
    effective_batch_size: int = config.SFTConfig.effective_batch_size,
    per_device_batch_size: int = 1,
    lora_r: int = config.SFTConfig().lora.r,
    lora_alpha: int = config.SFTConfig().lora.alpha,
    lora_dropout: float = config.SFTConfig().lora.dropout,
    max_length: int = 4096,
    seed: int = 0,
    dtype: str = "bfloat16",
) -> str:
    """Run the SFT finetune; save the adapter to `output_dir`. Returns the path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base_model = base_model or config.GEMMA_MODELS[config.PRIMARY_TARGET]
    if output_dir is None:
        output_dir = os.path.join(config.RESULTS_DIR, "training", f"sft_{variant}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=getattr(torch, dtype), device_map="auto")

    dataset = _build_sft_dataset(data_path, tokenizer)
    peft_config = build_lora_config(
        r=lora_r, alpha=lora_alpha, dropout=lora_dropout, layers=layers)

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        bf16=(dtype == "bfloat16"),
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        dataset_text_field="text",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
