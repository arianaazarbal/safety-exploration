"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4.1 / Table 9).

Hyperparameters: 1,150 samples (650 calm + 500 Dolci-Instruct-SFT), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections, effective
batch size 8. The paper finds SFT ineffective (and the 'teacher' variant
counter-productive); this trainer reproduces it for that comparison.
"""
from __future__ import annotations

import json
import os

from ..config import Config


def _load_samples(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def train_sft(cfg: Config, dataset_path: str, *,
              output_dir: str | None = None) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tcfg = cfg.training
    scfg = tcfg.sft
    base_model = tcfg.base_model
    output_dir = output_dir or os.path.join(cfg.run.output_dir, "models",
                                             "gemma-sft")
    os.makedirs(output_dir, exist_ok=True)

    samples = _load_samples(dataset_path)
    dataset = Dataset.from_list(samples)  # each row: {"messages": [...]}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    peft_config = LoraConfig(
        r=int(tcfg.lora.rank),
        lora_alpha=int(scfg.lora_alpha),
        target_modules=list(tcfg.lora.target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    batch = int(scfg.effective_batch_size)
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=int(scfg.epochs),
        learning_rate=float(scfg.learning_rate),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
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
