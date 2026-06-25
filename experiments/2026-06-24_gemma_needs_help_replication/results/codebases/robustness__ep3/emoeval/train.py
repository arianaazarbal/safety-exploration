"""LoRA fine-tuning: DPO and SFT (Section 4 / Appendix E).

Uses TRL's DPOTrainer / SFTTrainer with PEFT LoRA adapters. Hyperparameters come
from ``config.TrainConfig`` (Table 9). Supports restricting adapters to a subset
of decoder layers for the Appendix I layer-ablation study.
"""
from __future__ import annotations

import os
from typing import Optional

from .config import ADAPTER_DIR, MODELS, ModelSpec, TrainConfig
from .utils import read_jsonl


def _target_modules(cfg: TrainConfig, num_layers: int) -> list[str]:
    """Build the LoRA target-module list.

    Default: every projection on every layer (Appendix E). If
    ``cfg.layers_to_train`` is set, restrict to those decoder layers by using
    fully-qualified module names containing the layer index.
    """
    if cfg.layers_to_train is None:
        return list(cfg.lora_target_modules)
    targets: list[str] = []
    for layer in cfg.layers_to_train:
        for proj in cfg.lora_target_modules:
            # Gemma-3 decoder layers live under model.language_model.layers.<i>.
            targets.append(f"layers.{layer}.{_proj_path(proj)}")
    return targets


def _proj_path(proj: str) -> str:
    if proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"self_attn.{proj}"
    return f"mlp.{proj}"


def _lora_config(cfg: TrainConfig, num_layers: int):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_target_modules(cfg, num_layers),
    )


def _load_base(spec: ModelSpec):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    num_layers = model.config.num_hidden_layers
    return model, tok, num_layers


def _per_device_bs_and_accum(effective: int) -> tuple[int, int]:
    """Pick a small per-device batch and grad-accum to hit the effective size."""
    per_device = 1
    accum = max(1, effective // per_device)
    return per_device, accum


def train_dpo(cfg: TrainConfig, data_path: str, base_model: str,
              output_name: str = "dpo_gemma") -> str:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    spec = MODELS[base_model]
    model, tok, num_layers = _load_base(spec)
    records = read_jsonl(data_path)
    ds = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in records
    ])

    out_dir = os.path.join(ADAPTER_DIR, output_name)
    per_device, accum = _per_device_bs_and_accum(cfg.effective_batch_size)
    args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        beta=cfg.dpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        gradient_checkpointing=True,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(cfg, num_layers),
    )
    trainer.train()
    trainer.save_model(out_dir)
    return out_dir


def train_sft(cfg: TrainConfig, data_path: str, base_model: str,
              output_name: str = "sft_gemma") -> str:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    spec = MODELS[base_model]
    model, tok, num_layers = _load_base(spec)
    records = read_jsonl(data_path)
    ds = Dataset.from_list([{"messages": r["messages"]} for r in records])

    out_dir = os.path.join(ADAPTER_DIR, output_name)
    per_device, accum = _per_device_bs_and_accum(cfg.effective_batch_size)
    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        gradient_checkpointing=True,
        max_length=4096,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(cfg, num_layers),
    )
    trainer.train()
    trainer.save_model(out_dir)
    return out_dir
