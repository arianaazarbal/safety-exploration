"""LoRA finetuning of Gemma-3-27B-it via TRL (DPO and SFT).

Implements the Section 4.1 / Appendix E setup:
  * LoRA rank-64 adapters on all attention + MLP projection layers
  * DPO: 1 epoch, lr 5e-5, beta 0.1, 280 pairs
  * SFT: 2 epochs, lr 1e-4, alpha 128, 650 calm + 500 Dolci samples

The ``layer_subset`` option (Appendix I) restricts LoRA to a contiguous range of
decoder layers via PEFT's ``layers_to_transform``, so the layer-ablation study
can be reproduced.
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig

from .config import TrainConfig


def _lora_config(tc: TrainConfig, layers_to_transform=None) -> LoraConfig:
    return LoraConfig(
        r=tc.lora_rank,
        lora_alpha=tc.lora_alpha,
        target_modules=tc.target_modules,
        layers_to_transform=layers_to_transform,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _layers_to_transform(tc: TrainConfig):
    if tc.layer_subset is None:
        return None
    lo, hi = tc.layer_subset
    return list(range(lo, hi))


def _grad_accum(tc: TrainConfig, per_device_bs: int) -> int:
    return max(1, tc.effective_batch_size // per_device_bs)


def train_dpo(base_model_id: str, dataset_path: str | Path, output_dir: str | Path,
              tc: TrainConfig, per_device_batch_size: int = 1) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype="bfloat16",
                                                 device_map="auto")
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        beta=tc.dpo_beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(tc, per_device_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(tc, _layers_to_transform(tc)),
    )
    trainer.train()
    adapter_dir = Path(output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    return adapter_dir


def train_sft(base_model_id: str, dataset_path: str | Path, output_dir: str | Path,
              tc: TrainConfig, per_device_batch_size: int = 1) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype="bfloat16",
                                                 device_map="auto")
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(tc, per_device_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(tc, _layers_to_transform(tc)),
    )
    trainer.train()
    adapter_dir = Path(output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    return adapter_dir
