"""DPO and SFT finetuning of gemma-3-27b-it with LoRA (Section 4.1, Table 9).

Uses TRL (DPOTrainer / SFTTrainer) + PEFT LoRA rank-64 adapters on all attention
and MLP projections. The DPO config matches Table 9 (1 epoch, lr 5e-5, beta 0.1,
effective batch 8); SFT matches (2 epochs, lr 1e-4). ``layers`` restricts the
LoRA adapters to a contiguous layer range for the Appendix-I.1 ablation.
"""
from __future__ import annotations

from pathlib import Path

from ..config import load_training, output_path


def _lora_config(*, alpha: int, layers: tuple[int, int] | None):
    from peft import LoraConfig

    cfg = load_training()["lora"]
    kwargs = dict(
        r=cfg["r"],
        lora_alpha=alpha,
        target_modules=cfg["target_modules"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers is not None:
        lo, hi = layers
        kwargs["layers_to_transform"] = list(range(lo, hi))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _load_jsonl_dataset(path: Path):
    from datasets import load_dataset

    return load_dataset("json", data_files=str(path), split="train")


def _base_and_tokenizer(base_model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tokenizer


def train_dpo(
    *,
    layers: tuple[int, int] | None = None,
    out_name: str = "dpo/final",
    dataset_path: Path | None = None,
) -> str:
    from trl import DPOConfig, DPOTrainer

    tcfg = load_training()
    dpo = tcfg["dpo"]
    base_model = tcfg["base_model"]

    ds = _load_jsonl_dataset(dataset_path or Path(output_path("training", "dpo_dataset.jsonl")))
    model, tokenizer = _base_and_tokenizer(base_model)
    peft_cfg = _lora_config(alpha=dpo["lora_alpha"], layers=layers)
    out_dir = str(output_path("training", out_name))

    args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=dpo["epochs"],
        learning_rate=dpo["learning_rate"],
        beta=dpo["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=dpo["effective_batch_size"],
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(out_dir)
    return out_dir


def train_sft(*, out_name: str = "sft_diverse/final", dataset_path: Path | None = None) -> str:
    from trl import SFTConfig, SFTTrainer

    tcfg = load_training()
    sft = tcfg["sft"]
    base_model = tcfg["base_model"]

    ds = _load_jsonl_dataset(dataset_path or Path(output_path("training", "sft_dataset.jsonl")))
    model, tokenizer = _base_and_tokenizer(base_model)
    peft_cfg = _lora_config(alpha=sft["lora_alpha"], layers=None)
    out_dir = str(output_path("training", out_name))

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=sft["epochs"],
        learning_rate=sft["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=sft["effective_batch_size"],
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(out_dir)
    return out_dir
