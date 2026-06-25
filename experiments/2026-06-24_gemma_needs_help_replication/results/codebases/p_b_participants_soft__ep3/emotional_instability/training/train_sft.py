"""SFT finetuning of Gemma-3-27B-it with LoRA (Table 9 / Appendix E).

1,150 samples (650 calm + 500 Dolci), 2 epochs, lr 1e-4, LoRA rank 64 /
alpha 128 on all attn+MLP projection modules. Supports the Appendix-F "teacher"
variant via a system prompt injected at dataset-build time.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ..config import LORA_TARGET_MODULES, PATHS, SFT


def _load_messages_dataset(path: str):
    from datasets import Dataset

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def train_sft(
    dataset_path: Optional[str] = None,
    base_model: str = "google/gemma-3-27b-it",
    output_dir: Optional[str] = None,
    variant: str = "diverse",        # "diverse" | "teacher"
    load_in_4bit: bool = True,
    cfg=SFT,
):
    """Run SFT and save the LoRA adapter. `variant` only affects the output dir
    name; the teacher system prompt is baked into the dataset by build_datasets.
    """
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    dataset_path = dataset_path or os.path.join(PATHS.datasets, "sft", "sft_dataset.jsonl")
    output_dir = output_dir or os.path.join(PATHS.checkpoints, f"gemma27b_sft_{variant}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = _load_messages_dataset(dataset_path)

    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    per_device = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
