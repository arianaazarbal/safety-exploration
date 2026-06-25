"""LoRA fine-tuning of Gemma-3-27B-it via TRL (Appendix E, Table 9).

Both methods use rank-64 LoRA on all attention + MLP projections
(``q,k,v,o,gate,up,down``). Hyperparameters:

           | DPO        | SFT
  epochs   | 1          | 2
  lr       | 5e-5       | 1e-4
  alpha    | 64         | 128
  beta     | 0.1        | -
  eff. bs  | 8          | 8

``target_layers`` supports the Appendix I ablation (restricting adapters to a
subset of decoder layers, e.g. ``range(30, 36)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass
class TrainConfig:
    base_model_id: str = "google/gemma-3-27b-it"
    output_dir: str = "outputs/training/dpo"
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    learning_rate: float = 5e-5
    epochs: int = 1
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    dpo_beta: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 1536
    dtype: str = "bfloat16"
    # Appendix I ablation: restrict LoRA to these decoder layer indices.
    target_layers: Sequence[int] | None = None


def _build_lora_config(cfg: TrainConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.target_layers is not None:
        # PEFT supports restricting by layer index.
        kwargs["layers_to_transform"] = list(cfg.target_layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _grad_accum(cfg: TrainConfig) -> int:
    return max(1, cfg.effective_batch_size // cfg.per_device_batch_size)


def _load_base(cfg: TrainConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=getattr(torch, cfg.dtype),
        device_map="auto",
        attn_implementation="eager",
    )
    return model, tok


def train_dpo(pairs_path: str | Path, cfg: TrainConfig) -> str:
    """Run DPO; returns the saved adapter path."""
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = _load_base(cfg)
    ds = load_dataset("json", data_files=str(pairs_path), split="train")

    args = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(cfg),
        beta=cfg.dpo_beta,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        bf16=cfg.dtype == "bfloat16",
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_build_lora_config(cfg),
    )
    trainer.train()
    adapter_path = str(Path(cfg.output_dir) / "adapter")
    trainer.save_model(adapter_path)
    return adapter_path


def train_sft(dataset_path: str | Path, cfg: TrainConfig) -> str:
    """Run SFT; returns the saved adapter path."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = _load_base(cfg)
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(cfg),
        max_length=cfg.max_length,
        bf16=cfg.dtype == "bfloat16",
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_build_lora_config(cfg),
    )
    trainer.train()
    adapter_path = str(Path(cfg.output_dir) / "adapter")
    trainer.save_model(adapter_path)
    return adapter_path


def dpo_config(**overrides) -> TrainConfig:
    base = dict(
        output_dir="outputs/training/dpo",
        lora_alpha=64, learning_rate=5e-5, epochs=1, dpo_beta=0.1,
    )
    base.update(overrides)
    return TrainConfig(**base)


def sft_config(**overrides) -> TrainConfig:
    base = dict(
        output_dir="outputs/training/sft",
        lora_alpha=128, learning_rate=1e-4, epochs=2,
    )
    base.update(overrides)
    return TrainConfig(**base)
