"""LoRA DPO finetune of Gemma-3-27B-it (PAPER Section 4 / Table 9 / Appendix I).

Trains rank-64 LoRA adapters with the DPO objective (β=0.1, lr=5e-5, 1 epoch,
effective batch 8). ``layers`` restricts adapters to a subset of decoder layers
for the Appendix-I ablations. The result is a PEFT adapter directory that the
Gemma client can load via ``adapter_path`` for re-evaluation.
"""

from __future__ import annotations

import os
from typing import Optional

from .. import config
from ..utils.io import read_jsonl
from .lora import LayerSpec, build_lora_config


def _build_trl_dpo_dataset(pairs_path: str, tokenizer):
    """Convert DPO pair rows into a TRL-style {prompt, chosen, rejected} dataset
    of strings, rendering the prompt with Gemma's chat template."""
    from datasets import Dataset

    rows = list(read_jsonl(pairs_path))
    records = []
    for r in rows:
        prompt = tokenizer.apply_chat_template(
            r["prompt_messages"], tokenize=False, add_generation_prompt=True)
        records.append({"prompt": prompt, "chosen": r["chosen"],
                        "rejected": r["rejected"]})
    return Dataset.from_list(records)


def train_dpo(
    pairs_path: str,
    *,
    base_model: Optional[str] = None,
    output_dir: Optional[str] = None,
    layers: LayerSpec = None,
    epochs: int = config.DPOConfig.epochs,
    learning_rate: float = config.DPOConfig.learning_rate,
    beta: float = config.DPOConfig.beta,
    effective_batch_size: int = config.DPOConfig.effective_batch_size,
    per_device_batch_size: int = 1,
    lora_r: int = config.DPOConfig().lora.r,
    lora_alpha: int = config.DPOConfig().lora.alpha,
    lora_dropout: float = config.DPOConfig().lora.dropout,
    max_length: int = 4096,
    max_prompt_length: int = 3072,
    seed: int = 0,
    dtype: str = "bfloat16",
) -> str:
    """Run the DPO finetune and save the adapter to `output_dir`. Returns the
    adapter path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    base_model = base_model or config.GEMMA_MODELS[config.PRIMARY_TARGET]
    if output_dir is None:
        tag = "all" if layers is None else str(layers).replace(" ", "")
        output_dir = os.path.join(config.RESULTS_DIR, "training", f"dpo_{tag}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=getattr(torch, dtype), device_map="auto")

    dataset = _build_trl_dpo_dataset(pairs_path, tokenizer)
    peft_config = build_lora_config(
        r=lora_r, alpha=lora_alpha, dropout=lora_dropout, layers=layers)

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    dpo_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        bf16=(dtype == "bfloat16"),
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
