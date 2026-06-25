"""LoRA DPO / SFT finetuning of Gemma-3-27B-it (Table 9 hyperparameters).

Both methods apply LoRA (rank 64) to all attention + MLP projections
(q/k/v/o/gate/up/down). The `layers` argument supports the Appendix I ablation
that restricts adapters to a subset of layers (e.g. layers 30-35).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..config import ADAPTERS_DIR
from .data import load_jsonl

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
BASE_MODEL_ID = "google/gemma-3-27b-it"


def _lora_config(rank: int, alpha: int, layers: Optional[List[int]] = None):
    from peft import LoraConfig
    kw = dict(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
              task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES)
    if layers is not None:                       # Appendix I layer-subset ablation
        kw["layers_to_transform"] = list(layers)
    return LoraConfig(**kw)


def _load_base(load_in_4bit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kw)
    return model, tok


def train_dpo(pairs_path: Path, *, out_dir: Optional[Path] = None,
              epochs: int = 1, lr: float = 5e-5, beta: float = 0.1,
              rank: int = 64, alpha: int = 64, effective_batch_size: int = 8,
              per_device_batch_size: int = 1, layers: Optional[List[int]] = None,
              load_in_4bit: bool = False) -> Path:
    """DPO finetune (Table 9: 1 epoch, lr 5e-5, beta 0.1, LoRA r=64 a=64, eff bs 8)."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    out_dir = Path(out_dir or (ADAPTERS_DIR / "dpo"))
    records = load_jsonl(pairs_path)
    ds = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")}
                            for r in records])
    model, tok = _load_base(load_in_4bit)
    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = DPOConfig(
        output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=lr,
        beta=beta, per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum, logging_steps=10,
        bf16=True, save_strategy="no", report_to=[])
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok,
                         peft_config=_lora_config(rank, alpha, layers))
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir


def train_sft(data_path: Path, *, out_dir: Optional[Path] = None,
              epochs: int = 2, lr: float = 1e-4, rank: int = 64, alpha: int = 128,
              effective_batch_size: int = 8, per_device_batch_size: int = 1,
              load_in_4bit: bool = False) -> Path:
    """SFT finetune (Table 9: 2 epochs, lr 1e-4, LoRA r=64 a=128, eff bs 8)."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    out_dir = Path(out_dir or (ADAPTERS_DIR / "sft_diverse"))
    records = load_jsonl(data_path)
    ds = Dataset.from_list([{"messages": r["messages"]} for r in records])
    model, tok = _load_base(load_in_4bit)
    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum, logging_steps=10,
        bf16=True, save_strategy="no", report_to=[], max_length=2048,
        packing=False)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok,
                         peft_config=_lora_config(rank, alpha))
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir
