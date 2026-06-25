"""DPO and SFT finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Appendix E).

Hyperparameters (Table 9):
            DPO            SFT
  epochs    1              2
  lr        5e-5           1e-4
  LoRA r    64             64
  LoRA a    64             128
  batch     8 (effective)  8 (effective)
  beta      0.1            —

Adapters target all attention + MLP projections. The ``layers_to_transform``
field on LoraConfig restricts adapters to a contiguous layer range for the
Appendix I layer-ablation study (e.g. layers 30-35 only).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import DPOConfig, SFTConfig
from .build_datasets import _load


def _peft_config(lora):
    from peft import LoraConfig as PeftLoraConfig

    kwargs = dict(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        task_type="CAUSAL_LM",
    )
    if lora.layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(lora.layers_to_transform)
    return PeftLoraConfig(**kwargs)


def _load_base(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    return model, tok


def _grad_accum(effective_batch: int, per_device: int = 1) -> int:
    return max(1, effective_batch // per_device)


def train_dpo(dataset_path: str | Path, cfg: DPOConfig, *,
              out_dir: str | Path, per_device_batch: int = 1) -> str:
    """Train a DPO LoRA adapter; return the adapter output path."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = _load_base(cfg.base_model)
    rows = _load(dataset_path)

    # Render chat prompts to strings via the tokenizer's chat template.
    def _fmt(row):
        prompt = tok.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}

    ds = Dataset.from_list([_fmt(r) for r in rows])

    args = TRLDPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(cfg.effective_batch_size, per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_peft_config(cfg.lora),
    )
    trainer.train()
    adapter_path = str(out_dir / "adapter")
    trainer.model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    _write_meta(out_dir, "dpo", cfg)
    return adapter_path


def train_sft(dataset_path: str | Path, cfg: SFTConfig, *,
              out_dir: str | Path, per_device_batch: int = 1) -> str:
    """Train an SFT LoRA adapter; return the adapter output path."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = _load_base(cfg.base_model)
    rows = _load(dataset_path)

    def _fmt(row):
        return {"text": tok.apply_chat_template(row["messages"], tokenize=False)}

    ds = Dataset.from_list([_fmt(r) for r in rows])

    args = TRLSFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(cfg.effective_batch_size, per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_peft_config(cfg.lora),
    )
    trainer.train()
    adapter_path = str(out_dir / "adapter")
    trainer.model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    _write_meta(out_dir, f"sft_{cfg.variant}", cfg)
    return adapter_path


def _write_meta(out_dir: Path, method: str, cfg) -> None:
    meta = {"method": method, "config": cfg.__dict__.copy()}
    meta["config"]["lora"] = cfg.lora.__dict__
    with open(out_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
