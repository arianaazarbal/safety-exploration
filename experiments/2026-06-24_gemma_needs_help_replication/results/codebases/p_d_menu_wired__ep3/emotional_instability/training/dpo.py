"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4.1 / Table 9).

Hyperparameters: 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all
attention+MLP projections, effective batch size 8, beta 0.1.
"""
from __future__ import annotations

import json
import os

from ..config import Config


def _load_pairs(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def train_dpo(cfg: Config, pairs_path: str, *,
              output_dir: str | None = None) -> str:
    """Train and save the DPO LoRA adapter; return the adapter directory."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tcfg = cfg.training
    dcfg = tcfg.dpo
    base_model = tcfg.base_model
    output_dir = output_dir or os.path.join(cfg.run.output_dir, "models",
                                             "gemma-dpo")
    os.makedirs(output_dir, exist_ok=True)

    pairs = _load_pairs(pairs_path)
    dataset = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    peft_config = LoraConfig(
        r=int(tcfg.lora.rank),
        lora_alpha=int(dcfg.lora_alpha),
        target_modules=list(tcfg.lora.target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    batch = int(dcfg.effective_batch_size)
    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=int(dcfg.epochs),
        learning_rate=float(dcfg.learning_rate),
        beta=float(dcfg.beta),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
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
