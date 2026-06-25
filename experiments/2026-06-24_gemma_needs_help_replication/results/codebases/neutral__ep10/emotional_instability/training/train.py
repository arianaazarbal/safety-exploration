"""DPO and SFT LoRA finetuning of Gemma-3-27B-it (Section 4 / Appendix E).

Hyperparameters are read from config (Table 9). LoRA adapters target all
attention + MLP projections (Appendix E). A `layers` argument restricts the
adapters to a subset of decoder layers, which drives the Appendix I layer-
ablation experiment (e.g. layers 30-35 only).

Uses TRL's DPOTrainer / SFTTrainer with PEFT. Imports are lazy so the module
loads without a training stack present.
"""

from __future__ import annotations

import os
from typing import Optional

from .. import config


def _lora_config(rank: int, alpha: int, layers: Optional[list[int]] = None):
    """Build a PEFT LoraConfig. If `layers` is given, only attach adapters to
    those decoder layers (used for the layer-ablation study)."""
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.LORA_TARGET_MODULES,
    )
    if layers is not None:
        # PEFT supports restricting to specific layer indices.
        kwargs["layers_to_transform"] = list(layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _load_base(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = config.get_model(model_name)
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    return model, tok


def train_dpo(
    dataset_path: str, output_dir: str, *,
    model_name: str = config.TARGET_FINETUNE_MODEL,
    layers: Optional[list[int]] = None,
    cfg: config.DPOConfig = config.DPO_CFG,
) -> str:
    """Run single-epoch DPO and save the LoRA adapter to `output_dir`."""
    from datasets import load_dataset
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    model, tok = _load_base(model_name)
    ds = load_dataset("json", data_files=dataset_path, split="train")

    # TRL expects prompt/chosen/rejected. Our prompt is a chat-message list;
    # render it with the tokenizer's chat template into a single string.
    def format_row(row):
        prompt = tok.apply_chat_template(row["prompt"], tokenize=False,
                                         add_generation_prompt=True)
        return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}

    ds = ds.map(format_row)

    args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(cfg.lora_rank, cfg.lora_alpha, layers),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    return output_dir


def train_sft(
    dataset_path: str, output_dir: str, *,
    model_name: str = config.TARGET_FINETUNE_MODEL,
    layers: Optional[list[int]] = None,
    cfg: config.SFTConfig = config.SFT_CFG,
) -> str:
    """Run SFT (2 epochs) on the calm + instruct-mix dataset."""
    from datasets import load_dataset
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    model, tok = _load_base(model_name)
    ds = load_dataset("json", data_files=dataset_path, split="train")

    args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(cfg.lora_rank, cfg.lora_alpha, layers),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    return output_dir
