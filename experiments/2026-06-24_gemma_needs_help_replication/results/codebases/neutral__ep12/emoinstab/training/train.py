"""LoRA DPO / SFT training of Gemma-3-27B-it (Section 4.1, Appendix E, Table 9).

Hyperparameters (Table 9):
              DPO            SFT
  dataset     280 pairs      1,150 samples
  epochs      1              2
  lr          5e-5           1e-4
  LoRA rank   64             64
  LoRA alpha  64             128
  eff. batch  8              8
  DPO beta    0.1            -

LoRA adapters target all attention + MLP projections
(q/k/v/o_proj, gate/up/down_proj). The Appendix-I layer ablation is supported
via `layers_to_transform` (e.g. layers 30-35 only).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .. import config

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

BASE_MODEL = "google/gemma-3-27b-it"


@dataclass
class TrainConfig:
    method: str                     # 'dpo' | 'sft'
    output_dir: str
    dataset_path: str
    epochs: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    beta: float = 0.1               # DPO only
    max_length: int = 4096
    layers_to_transform: Optional[List[int]] = None  # Appendix I ablation
    base_model: str = BASE_MODEL


def dpo_config(dataset_path: str, output_dir: str,
               layers: Optional[List[int]] = None) -> TrainConfig:
    return TrainConfig(
        method="dpo", output_dir=output_dir, dataset_path=dataset_path,
        epochs=1, learning_rate=5e-5, lora_rank=64, lora_alpha=64, beta=0.1,
        layers_to_transform=layers)


def sft_config(dataset_path: str, output_dir: str) -> TrainConfig:
    return TrainConfig(
        method="sft", output_dir=output_dir, dataset_path=dataset_path,
        epochs=2, learning_rate=1e-4, lora_rank=64, lora_alpha=128)


def _peft_config(cfg: TrainConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    if cfg.layers_to_transform is not None:
        kwargs["layers_to_transform"] = cfg.layers_to_transform
    return LoraConfig(**kwargs)


def _grad_accum(cfg: TrainConfig) -> int:
    return max(1, cfg.effective_batch_size // cfg.per_device_batch_size)


def _load_jsonl(path: str) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _render_prompt(tokenizer, messages: List[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def train(cfg: TrainConfig):
    """Run training. Imports heavy deps lazily so the module loads without a GPU."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto")

    peft_cfg = _peft_config(cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if cfg.method == "dpo":
        from trl import DPOConfig, DPOTrainer

        rows = _load_jsonl(cfg.dataset_path)
        data = Dataset.from_list([
            {"prompt": _render_prompt(tokenizer, r["prompt_messages"]),
             "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in rows
        ])
        args = DPOConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=_grad_accum(cfg),
            beta=cfg.beta,
            max_length=cfg.max_length,
            max_prompt_length=cfg.max_length // 2,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
        )
        trainer = DPOTrainer(model=model, args=args, train_dataset=data,
                             processing_class=tokenizer, peft_config=peft_cfg)

    elif cfg.method == "sft":
        from trl import SFTConfig, SFTTrainer

        rows = _load_jsonl(cfg.dataset_path)
        data = Dataset.from_list([
            {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)}
            for r in rows
        ])
        args = SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=_grad_accum(cfg),
            max_seq_length=cfg.max_length,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            dataset_text_field="text",
        )
        trainer = SFTTrainer(model=model, args=args, train_dataset=data,
                             processing_class=tokenizer, peft_config=peft_cfg)
    else:
        raise ValueError(f"Unknown method {cfg.method}")

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[train:{cfg.method}] saved adapter -> {cfg.output_dir}")
    return cfg.output_dir
