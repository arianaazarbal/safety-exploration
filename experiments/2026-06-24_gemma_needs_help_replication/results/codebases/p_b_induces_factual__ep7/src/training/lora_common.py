"""Shared LoRA / model-loading helpers for the DPO and SFT trainers."""
from __future__ import annotations

from typing import Optional

import config


def load_base_for_training(model_id: str, *, load_in_4bit: bool = True):
    """Load the Gemma base model + tokenizer for LoRA fine-tuning.

    4-bit (QLoRA) is the default so the 27B model fits on a single ~48GB GPU; pass
    ``load_in_4bit=False`` for full-precision LoRA on larger hardware.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tokenizer


def make_lora_config(*, rank: int, alpha: int, layer_ablation: str = "all"):
    """Build a peft LoraConfig over all attention+MLP projections (Appendix E),
    optionally restricted to a layer range for the Section 4.2 / Appendix I ablation."""
    from peft import LoraConfig

    layers = config.LORA_LAYER_ABLATIONS.get(layer_ablation)
    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.LORA_TARGET_MODULES,
    )
    if layers is not None:
        # Restrict adapters to specific decoder layers (e.g. 30-35 only).
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
