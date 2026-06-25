"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

TRL ``DPOTrainer`` + PEFT rank-64 LoRA on all attention/MLP projections, 1 epoch,
lr 5e-5, beta 0.1, effective batch size 8 (Table 9). An optional ``layer_range``
restricts LoRA to a contiguous block of decoder layers for the Appendix I.1
layer-ablation study.

Heavy imports (`torch`, `transformers`, `trl`, `peft`) are deferred to runtime.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

from .. import config


def _lora_config(layer_range: Optional[Tuple[int, int]],
                 rank: int, alpha: int):
    from peft import LoraConfig
    kwargs = dict(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=config.LORA_TARGET_MODULES,
    )
    if layer_range is not None:
        lo, hi = layer_range
        kwargs["layers_to_transform"] = list(range(lo, hi + 1))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def train_dpo(
    pairs_path: str,
    output_dir: str,
    *,
    base_model_key: str = "gemma-3-27b-it",
    layer_range: Optional[Tuple[int, int]] = None,
    per_device_batch_size: int = 1,
    seed: int = 0,
) -> str:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = config.DPO
    hf_id = config.GEMMA_MODELS[base_model_key]
    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")

    dataset = load_dataset("json", data_files=pairs_path, split="train")

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(layer_range, cfg.lora_rank, cfg.lora_alpha),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
