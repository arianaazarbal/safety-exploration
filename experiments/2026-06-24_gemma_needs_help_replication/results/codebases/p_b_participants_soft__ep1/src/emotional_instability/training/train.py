"""LoRA SFT and DPO training of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters (Table 9):
                       DPO            SFT
    Dataset size       280 pairs      1,150 samples
    Epochs             1              2
    Learning rate      5e-5           1e-4
    LoRA rank          64             64
    LoRA alpha         64             128
    Effective batch    8              8
    DPO beta           0.1            -

LoRA adapters are applied to all attention + MLP projections
(q/k/v/o_proj, gate/up/down_proj). The Appendix I layer-subset ablation is
supported via ``layers_to_tune`` (e.g. ``range(30, 36)`` for "layers 30-35
only").

Heavy imports (torch/transformers/peft/trl) are deferred to call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import config

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass
class LoRASettings:
    rank: int = 64
    alpha: int = 64
    dropout: float = 0.0
    target_modules: tuple[str, ...] = tuple(LORA_TARGET_MODULES)
    # Appendix I: restrict adapters to a subset of decoder layers.
    layers_to_tune: tuple[int, ...] | None = None


def _lora_config(settings: LoRASettings):
    from peft import LoraConfig

    kwargs = dict(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if settings.layers_to_tune is not None:
        kwargs["layers_to_transform"] = list(settings.layers_to_tune)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _load_base(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tokenizer


def _grad_accum(effective_batch: int, per_device_batch: int) -> int:
    return max(1, effective_batch // per_device_batch)


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #

def train_dpo(
    dpo_jsonl: Path,
    output_dir: Path,
    *,
    base_model_id: str = config.PARTICIPANTS[config.SOURCE_MODEL].model_id,
    lora: LoRASettings | None = None,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    effective_batch: int = 8,
    per_device_batch: int = 1,
) -> Path:
    """Single-epoch LoRA DPO (Section 4.1 / Table 9). Returns adapter dir."""
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    lora = lora or LoRASettings(rank=64, alpha=64)
    model, tokenizer = _load_base(base_model_id)
    dataset = load_dataset("json", data_files=str(dpo_jsonl), split="train")

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(effective_batch, per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(lora),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #

def train_sft(
    sft_jsonl: Path,
    output_dir: Path,
    *,
    base_model_id: str = config.PARTICIPANTS[config.SOURCE_MODEL].model_id,
    lora: LoRASettings | None = None,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    effective_batch: int = 8,
    per_device_batch: int = 1,
    system_prompt: str | None = None,
) -> Path:
    """Two-epoch LoRA SFT (Section 4.1 / Table 9). ``system_prompt`` injects the
    'teacher' variant (Appendix F) when set. Returns adapter dir."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    lora = lora or LoRASettings(rank=64, alpha=128)
    model, tokenizer = _load_base(base_model_id)
    dataset = load_dataset("json", data_files=str(sft_jsonl), split="train")

    if system_prompt is not None:
        def _prepend_system(example):
            msgs = example["messages"]
            if not msgs or msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": system_prompt}] + msgs
            return {"messages": msgs}

        dataset = dataset.map(_prepend_system)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(effective_batch, per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(lora),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir
