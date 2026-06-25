"""DPO finetuning of Gemma-3-27B-it (Section 4 / Table 9).

1 epoch, LR 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attention+MLP
projections, effective batch size 8. Trains on the 280 preference pairs built
by build_dpo.py and writes a LoRA adapter to outputs/finetune/dpo.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from .lora_utils import build_lora_config


def _format_dataset(pairs_path: Path, tokenizer):
    from datasets import Dataset

    rows = [json.loads(line) for line in open(pairs_path)]
    data = {"prompt": [], "chosen": [], "rejected": []}
    for r in rows:
        prompt = tokenizer.apply_chat_template(
            r["prompt_messages"], tokenize=False, add_generation_prompt=True)
        data["prompt"].append(prompt)
        data["chosen"].append(r["chosen"])
        data["rejected"].append(r["rejected"])
    return Dataset.from_dict(data)


def train(cfg: Config, pairs_path: Path | None = None, out_dir: Path | None = None) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dcfg = cfg["finetune"]["dpo"]
    base = cfg["finetune"]["base_model"]
    pairs_path = pairs_path or (cfg.path_for("finetune") / "dpo_pairs.jsonl")
    out_dir = out_dir or (cfg.path_for("finetune") / "dpo")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    peft_config = build_lora_config(cfg, dcfg["lora_rank"], dcfg["lora_alpha"])
    dataset = _format_dataset(pairs_path, tokenizer)

    bs = dcfg["effective_batch_size"]
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=dcfg["epochs"],
        learning_rate=dcfg["learning_rate"],
        beta=dcfg["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=bs,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        max_length=2048,
        max_prompt_length=1536,
        report_to=[],
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=dataset,
                         processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
