"""LoRA DPO and SFT training of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters from Table 9:
  DPO : 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64, beta 0.1, eff. bs 8
  SFT : 1150 samples, 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, eff. bs 8
  LoRA on all attention + MLP projection layers
  (q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)

The Appendix I layer-subset ablation is supported via `lora_layers` (a set of
layer indices to which adapters are restricted).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def _lora_config(rank: int, alpha: int, layers: Optional[list[int]] = None):
    from peft import LoraConfig
    kwargs = dict(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES)
    if layers is not None:
        # restrict adapters to specific decoder layers (Appendix I ablation)
        kwargs["layers_to_transform"] = list(layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _load_base(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    return model, tok


def train_dpo(dpo_jsonl: Path, *, model_id: str = config.FINETUNE_BASE.model_id,
              out_dir: Optional[Path] = None, rank: int = 64, alpha: int = 64,
              beta: float = 0.1, lr: float = 5e-5, epochs: int = 1,
              batch_size: int = 1, grad_accum: int = 8,
              lora_layers: Optional[list[int]] = None) -> Path:
    """One-epoch LoRA DPO. Returns the adapter directory."""
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    out_dir = out_dir or (config.ADAPTERS_DIR / "dpo")
    model, tok = _load_base(model_id)
    ds = load_dataset("json", data_files=str(dpo_jsonl), split="train")

    args = DPOConfig(
        output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, beta=beta,
        logging_steps=10, save_strategy="epoch", bf16=True,
        gradient_checkpointing=True, report_to=[])
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        peft_config=_lora_config(rank, alpha, lora_layers))
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


def train_sft(sft_jsonl: Path, *, model_id: str = config.FINETUNE_BASE.model_id,
              out_dir: Optional[Path] = None, rank: int = 64, alpha: int = 128,
              lr: float = 1e-4, epochs: int = 2, batch_size: int = 1,
              grad_accum: int = 8) -> Path:
    """Two-epoch LoRA SFT on calm+Dolci data. Returns the adapter directory."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    out_dir = out_dir or (config.ADAPTERS_DIR / "sft")
    model, tok = _load_base(model_id)
    ds = load_dataset("json", data_files=str(sft_jsonl), split="train")

    args = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, logging_steps=10,
        save_strategy="epoch", bf16=True, gradient_checkpointing=True,
        report_to=[], packing=False)
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        peft_config=_lora_config(rank, alpha))
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir
