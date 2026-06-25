"""Section 4.1 / Appendix E: LoRA SFT and DPO finetuning of Gemma-3-27B-it.

Hyperparameters (Table 9):
                    DPO            SFT
    Dataset size    280 pairs      1,150 samples
    Epochs          1              2
    Learning rate   5e-5           1e-4
    LoRA rank       64             64
    LoRA alpha      64             128
    Eff. batch      8              8
    DPO beta        0.1            -

LoRA adapters are applied to all attention + MLP projections (q/k/v/o_proj,
gate/up/down_proj). `layers` lets us restrict adapters to a subset for the
Appendix I layer-ablation study.

Uses TRL (SFTTrainer / DPOTrainer) + PEFT. Imports are local so the module can
be inspected without a training stack installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
GEMMA_27B_IT = "google/gemma-3-27b-it"


@dataclass
class TrainConfig:
    method: str                       # "sft" | "dpo"
    dataset_path: Path
    output_dir: Path
    epochs: int
    learning_rate: float
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    dpo_beta: float = 0.1
    max_seq_len: int = 4096
    layers: Optional[list[int]] = None  # restrict LoRA to these layer indices
    seed: int = config.GLOBAL_SEED


def dpo_config(dataset_path: Path, output_dir: Path,
               layers: Optional[list[int]] = None) -> TrainConfig:
    return TrainConfig("dpo", dataset_path, output_dir, epochs=1,
                       learning_rate=5e-5, lora_alpha=64, dpo_beta=0.1,
                       layers=layers)


def sft_config(dataset_path: Path, output_dir: Path) -> TrainConfig:
    return TrainConfig("sft", dataset_path, output_dir, epochs=2,
                       learning_rate=1e-4, lora_alpha=128)


# --------------------------------------------------------------------------- #
# LoRA / model setup
# --------------------------------------------------------------------------- #
def _target_modules(cfg: TrainConfig) -> list[str]:
    """Return LoRA target module patterns, optionally restricted to layers.

    For the Appendix I ablation we scope adapters to specific decoder layers by
    fully-qualifying the module names (e.g. ``model.layers.30.self_attn.q_proj``).
    """
    if cfg.layers is None:
        return LORA_TARGET_MODULES
    targets = []
    for layer in cfg.layers:
        for mod in LORA_TARGET_MODULES:
            targets.append(f"model.layers.{layer}.*{mod}")
    return targets


def _load_base():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GEMMA_27B_IT)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_27B_IT, torch_dtype=torch.bfloat16, device_map="auto",
    )
    return model, tok


def _peft_config(cfg: TrainConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_target_modules(cfg),
    )


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def train_sft(cfg: TrainConfig) -> Path:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = _load_base()
    ds = load_dataset("json", data_files=str(cfg.dataset_path), split="train")

    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)
    args = SFTConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=cfg.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=_peft_config(cfg),
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    return cfg.output_dir


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def _format_dpo_example(ex: dict, tok) -> dict:
    """Render a preference pair into TRL's prompt/chosen/rejected text form."""
    prompt_text = tok.apply_chat_template(
        ex["prompt"], tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt_text, "chosen": ex["chosen"],
            "rejected": ex["rejected"]}


def train_dpo(cfg: TrainConfig) -> Path:
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = _load_base()
    ds = load_dataset("json", data_files=str(cfg.dataset_path), split="train")
    ds = ds.map(lambda ex: _format_dpo_example(ex, tok),
                remove_columns=ds.column_names)

    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)
    args = DPOConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.dpo_beta,
        max_length=cfg.max_seq_len,
        max_prompt_length=cfg.max_seq_len // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=_peft_config(cfg),
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    return cfg.output_dir


def train(cfg: TrainConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "train_config.json").write_text(json.dumps({
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in cfg.__dict__.items()
    }, indent=2))
    return train_dpo(cfg) if cfg.method == "dpo" else train_sft(cfg)
