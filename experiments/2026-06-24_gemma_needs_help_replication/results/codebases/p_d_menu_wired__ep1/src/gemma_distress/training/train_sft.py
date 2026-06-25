"""SFT finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 Dolci), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128 on all projections, effective batch size 8.

The paper reports SFT is ineffective (and the 'teacher' variant can increase
frustration); we implement it faithfully so that result can be reproduced.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Config


def _load_sft_dataset(path: str | Path):
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def train_sft(cfg: Config, dataset_path: str | Path, output_dir: str | Path | None = None) -> str:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tcfg = cfg.training
    scfg = tcfg["sft"]
    base_model = tcfg["base_model"]
    out_dir = str(output_dir or (cfg.output_dir / "adapters" / "sft"))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    ds = _load_sft_dataset(dataset_path)

    def _to_text(row: dict[str, Any]) -> dict[str, str]:
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(_to_text, remove_columns=ds.column_names)

    peft_config = LoraConfig(
        r=tcfg["lora_rank"],
        lora_alpha=scfg["lora_alpha"],
        target_modules=tcfg["lora_target_modules"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    bs = scfg["effective_batch_size"]
    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=scfg["epochs"],
        learning_rate=scfg["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=bs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
