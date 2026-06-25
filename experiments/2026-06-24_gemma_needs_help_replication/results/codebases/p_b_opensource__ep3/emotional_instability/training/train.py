"""LoRA SFT and DPO trainers for Gemma-3-27B-it (Section 4.1, Appendix E).

Both use rank-64 LoRA adapters on all attention + MLP projections (Table 9).
The ``layers_to_transform`` argument restricts the adapter to a contiguous band
of decoder layers — used for the Appendix-I localisation ablation (e.g. layers
30-35 only, or layer 40 onwards).

These functions wrap TRL's ``SFTTrainer`` / ``DPOTrainer`` and PEFT's
``LoraConfig``. They are written against the TRL >= 0.9 / PEFT >= 0.11 APIs
listed in ``requirements.txt``; see DESIGN.md for the version-drift caveat.
Hyperparameters (epochs, LR, beta, batch size) come from ``config`` so the
paper's exact values are auditable in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import config

from .. import storage

GEMMA_27B_IT = "google/gemma-3-27b-it"


def _lora_config(
    *,
    lora_alpha: int,
    layers_to_transform: Sequence[int] | None = None,
):
    from peft import LoraConfig

    kwargs = dict(
        r=config.LORA.r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.LORA.target_modules),
    )
    if layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(layers_to_transform)
    return LoraConfig(**kwargs)


def _load_base(model_id: str = GEMMA_27B_IT, *, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto")
    return model, tok


def _to_dataset(records: list[dict]):
    from datasets import Dataset
    return Dataset.from_list(records)


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def train_sft(
    examples: list[dict],
    output_dir: str | Path,
    *,
    epochs: int = config.SFT.epochs,
    learning_rate: float = config.SFT.learning_rate,
    lora_alpha: int = config.SFT.lora_alpha,
    effective_batch_size: int = config.SFT.effective_batch_size,
    per_device_batch_size: int = 1,
    layers_to_transform: Sequence[int] | None = None,
    base_model: str = GEMMA_27B_IT,
) -> Path:
    """Train a LoRA SFT adapter on conversational ``{"messages": [...]}`` examples.

    Loss is computed on assistant turns only (``assistant_only_loss``) so the
    model is trained to produce the calm responses, not to parrot the prompts.
    Returns the adapter output directory.
    """
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)
    model, tok = _load_base(base_model)
    grad_accum = max(1, effective_batch_size // per_device_batch_size)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        assistant_only_loss=True,   # mask user/system tokens in the loss
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=_to_dataset(examples),
        processing_class=tok,
        peft_config=_lora_config(lora_alpha=lora_alpha,
                                 layers_to_transform=layers_to_transform),
    )
    if not examples:
        raise ValueError("SFT example set is empty; generate calm data first.")
    trainer.train()
    trainer.save_model(str(output_dir))
    storage.write_json(output_dir / "train_meta.json", {
        "kind": "sft", "n_examples": len(examples), "epochs": epochs,
        "learning_rate": learning_rate, "lora_alpha": lora_alpha,
        "lora_r": config.LORA.r, "base_model": base_model,
        "layers_to_transform": list(layers_to_transform) if layers_to_transform else None,
    })
    return output_dir


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def train_dpo(
    pairs: list[dict],
    output_dir: str | Path,
    *,
    epochs: int = config.DPO.epochs,
    learning_rate: float = config.DPO.learning_rate,
    beta: float = config.DPO.beta,
    lora_alpha: int = config.DPO.lora_alpha,
    effective_batch_size: int = config.DPO.effective_batch_size,
    per_device_batch_size: int = 1,
    layers_to_transform: Sequence[int] | None = None,
    base_model: str = GEMMA_27B_IT,
) -> Path:
    """Train a LoRA DPO adapter on ``{"prompt","chosen","rejected"}`` pairs.

    ``prompt`` is conversational (a list of messages); ``chosen``/``rejected``
    are assistant-turn strings. Returns the adapter output directory.
    """
    from trl import DPOConfig, DPOTrainer

    output_dir = Path(output_dir)
    model, tok = _load_base(base_model)
    grad_accum = max(1, effective_batch_size // per_device_batch_size)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    if not pairs:
        raise ValueError("DPO pair set is empty; generate calm + frustrated data first.")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=_to_dataset(pairs),
        processing_class=tok,
        peft_config=_lora_config(lora_alpha=lora_alpha,
                                 layers_to_transform=layers_to_transform),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    storage.write_json(output_dir / "train_meta.json", {
        "kind": "dpo", "n_pairs": len(pairs), "epochs": epochs,
        "learning_rate": learning_rate, "beta": beta, "lora_alpha": lora_alpha,
        "lora_r": config.LORA.r, "base_model": base_model,
        "layers_to_transform": list(layers_to_transform) if layers_to_transform else None,
    })
    return output_dir
