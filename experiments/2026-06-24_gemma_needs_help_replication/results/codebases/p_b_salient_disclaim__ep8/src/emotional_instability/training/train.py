"""DPO and SFT training of Gemma-3-27B-it with LoRA (Section 4.1, Table 9).

Uses TRL's DPOTrainer / SFTTrainer with a PEFT LoRA config. The `lora_layers`
config option maps to PEFT's `layers_to_transform`, which is exactly what the
Appendix I layer-subset ablations need (e.g. adapters on layers 30-35 only).

Heavy imports are local so importing this module is cheap.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..config import ModelRegistry, load_training_config


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _lora_config(cfg: dict):
    from peft import LoraConfig

    kwargs = dict(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.0),
        target_modules=cfg["lora_target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    if cfg.get("lora_layers"):  # restrict to a subset of decoder layers
        kwargs["layers_to_transform"] = list(cfg["lora_layers"])
    return LoraConfig(**kwargs)


def _load_base_for_training(registry: ModelRegistry, base_model: str):
    """Load the base instruct weights + tokenizer for finetuning (no adapter)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = registry.spec(base_model)
    hf_id = spec.get("hf_id")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[spec.get("dtype", "bfloat16")]
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")
    return model, tok


def _grad_accum(effective_batch: int, per_device: int) -> int:
    return max(1, effective_batch // per_device)


def train_dpo(
    dataset_path: Path,
    training_cfg: Optional[dict] = None,
    registry: Optional[ModelRegistry] = None,
    per_device_batch: int = 1,
    output_dir: Optional[str] = None,
) -> str:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    training_cfg = training_cfg or load_training_config()
    registry = registry or ModelRegistry()
    cfg = training_cfg["dpo"]
    out_dir = output_dir or cfg["output_dir"]

    model, tok = _load_base_for_training(registry, training_cfg["base_model"])
    peft_cfg = _lora_config(cfg)

    rows = _read_jsonl(dataset_path)
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])

    args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        beta=cfg["beta"],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(cfg["effective_batch_size"], per_device_batch),
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    adapter_dir = str(Path(out_dir) / "adapter")
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"Saved DPO LoRA adapter to {adapter_dir}")
    return adapter_dir


def train_sft(
    dataset_path: Path,
    training_cfg: Optional[dict] = None,
    registry: Optional[ModelRegistry] = None,
    per_device_batch: int = 1,
    output_dir: Optional[str] = None,
) -> str:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    training_cfg = training_cfg or load_training_config()
    registry = registry or ModelRegistry()
    cfg = training_cfg["sft"]
    out_dir = output_dir or cfg["output_dir"]

    model, tok = _load_base_for_training(registry, training_cfg["base_model"])
    peft_cfg = _lora_config(cfg)

    rows = _read_jsonl(dataset_path)
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(cfg["effective_batch_size"], per_device_batch),
        max_length=cfg["max_length"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    adapter_dir = str(Path(out_dir) / "adapter")
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"Saved SFT LoRA adapter to {adapter_dir}")
    return adapter_dir
