"""LoRA DPO / SFT training of Gemma-3-27B-it (Section 4.1, Appendix E, Table 9).

Both methods use rank-64 LoRA adapters on all attention and MLP projection layers, an
effective batch size of 8, and the learning rates / epochs from Table 9 (DPO: 1 epoch,
5e-5, beta 0.1; SFT: 2 epochs, 1e-4, alpha 128). The Appendix I layer-ablation experiment
is supported via ``cfg.training.lora_layer_range``: when set, LoRA adapters are applied
only to that half-open range of decoder layers (e.g. ``[30, 35]``).

TRL's ``DPOTrainer`` / ``SFTTrainer`` consume the conversational datasets built by
:mod:`build_datasets`; with a PEFT config and no explicit reference model, DPO uses the
base model (adapters disabled) as the implicit reference.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..config import Config

logger = logging.getLogger(__name__)


def _grad_accum(cfg: Config) -> int:
    eff = cfg.training.effective_batch_size
    per = cfg.training.per_device_batch_size
    return max(1, eff // per)


def build_lora_config(cfg: Config, *, alpha: int):
    """Construct the PEFT LoRA config, honouring an optional layer-range restriction."""
    from peft import LoraConfig

    kwargs = dict(
        r=cfg.training.lora_rank,
        lora_alpha=alpha,
        target_modules=cfg.training.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.training.lora_layer_range is not None:
        start, end = cfg.training.lora_layer_range
        kwargs["layers_to_transform"] = list(range(start, end))
        # Gemma decoder layers are addressed as model.layers.<i>...
        kwargs["layers_pattern"] = "layers"
        logger.info("Restricting LoRA to decoder layers [%d, %d)", start, end)
    return LoraConfig(**kwargs)


def _load_base(cfg: Config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.training.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.training.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
    except (ValueError, KeyError, OSError):
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            cfg.training.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
    return model, tok


def train_dpo(cfg: Config, dpo_jsonl: str, output_dir: str) -> str:
    """Train a DPO LoRA adapter; returns the adapter directory."""
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = _load_base(cfg)
    lora = build_lora_config(cfg, alpha=cfg.training.lora_alpha_dpo)
    dataset = load_dataset("json", data_files=dpo_jsonl, split="train")

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.training.dpo_epochs,
        learning_rate=cfg.training.dpo_learning_rate,
        beta=cfg.training.dpo_beta,
        per_device_train_batch_size=cfg.training.per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(cfg),
        max_length=cfg.training.max_seq_len,
        max_prompt_length=cfg.training.max_seq_len // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # implicit reference: base model with adapters disabled
        args=args,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Saved DPO adapter to %s", output_dir)
    return output_dir


def train_sft(cfg: Config, sft_jsonl: str, output_dir: str) -> str:
    """Train an SFT LoRA adapter; returns the adapter directory."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = _load_base(cfg)
    lora = build_lora_config(cfg, alpha=cfg.training.lora_alpha_sft)
    dataset = load_dataset("json", data_files=sft_jsonl, split="train")

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.training.sft_epochs,
        learning_rate=cfg.training.sft_learning_rate,
        per_device_train_batch_size=cfg.training.per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(cfg),
        max_length=cfg.training.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Saved SFT adapter to %s", output_dir)
    return output_dir
