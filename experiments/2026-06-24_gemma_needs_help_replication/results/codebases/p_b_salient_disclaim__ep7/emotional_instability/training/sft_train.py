"""SFT finetuning of Gemma-3-27B-it with LoRA (Table 9, Appendix E).

Hyperparameters (config.SFT): 1150 samples (650 calm + 500 Dolci), 2 epochs,
lr 1e-4, LoRA rank 64, alpha 128, effective batch size 8, adapters on all
attention + MLP projections.

Used for both the "diverse" and "teacher" SFT models — the difference is only in
the calm-data source (see calm_data.generate_calm_responses).
"""

from __future__ import annotations

import os
from typing import Optional

import config
from .calm_data import SFTExample


def build_lora_config():
    from peft import LoraConfig
    return LoraConfig(
        r=config.SFT.lora_rank,
        lora_alpha=config.SFT.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.LORA_TARGET_MODULES,
    )


def train_sft(examples: list[SFTExample], *,
              base_model: str = "google/gemma-3-27b-it",
              output_dir: str = None,
              seed: int = config.SEED):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    output_dir = output_dir or os.path.join(config.OUTPUT_DIR, "sft-adapter")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TRL's SFTTrainer accepts a "messages" column and applies the chat template,
    # training only on the assistant completions when a chat template is present.
    ds = Dataset.from_list([{"messages": ex.messages} for ex in examples])

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        config.TORCH_DTYPE, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=config.DEVICE_MAP)

    per_device_bs = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "1"))
    grad_accum = max(1, config.SFT.effective_batch_size // per_device_bs)

    args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        bf16=(config.TORCH_DTYPE == "bfloat16"),
        fp16=(config.TORCH_DTYPE == "float16"),
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to=[],
        # train on completions only (mask the prompt tokens)
        assistant_only_loss=True,
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
