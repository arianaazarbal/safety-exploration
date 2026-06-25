"""SFT finetuning of Gemma-3-27B-it (Section 4 / Table 9).

2 epochs, LR 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections,
effective batch size 8. Trains on the calm+instruct mixture from build_sft.py.
Per Section 4.2 this baseline is *expected to fail* to reduce distress; we
implement it to reproduce that negative result and the SFT-vs-DPO comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from .lora_utils import build_lora_config


def _format_dataset(sft_path: Path):
    from datasets import Dataset

    rows = [json.loads(line) for line in open(sft_path)]
    return Dataset.from_dict({"messages": [r["messages"] for r in rows]})


def train(cfg: Config, variant: str = "diverse", sft_path: Path | None = None,
          out_dir: Path | None = None) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    scfg = cfg["finetune"]["sft"]
    base = cfg["finetune"]["base_model"]
    sft_path = sft_path or (cfg.path_for("finetune") / f"sft_{variant}.jsonl")
    out_dir = out_dir or (cfg.path_for("finetune") / f"sft_{variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    peft_config = build_lora_config(cfg, scfg["lora_rank"], scfg["lora_alpha"])
    dataset = _format_dataset(sft_path)

    bs = scfg["effective_batch_size"]
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=scfg["epochs"],
        learning_rate=scfg["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=bs,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        max_length=2048,
        report_to=[],
        packing=False,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                         processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
