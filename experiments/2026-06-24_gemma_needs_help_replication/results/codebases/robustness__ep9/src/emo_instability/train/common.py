"""Shared training utilities: model/tokenizer loading and LoRA config."""
from __future__ import annotations

from ..config import LoRAConfig, get_model


def load_base_for_training(model_key: str, *, load_in_4bit: bool = False):
    """Load a Gemma instruct model + tokenizer for finetuning."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_model(model_key)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, spec


def make_lora_config(lora: LoRAConfig, alpha: int):
    """Build a PEFT LoraConfig, optionally restricted to a subset of layers
    (Appendix I layer ablations) via ``lora.layers_to_transform``."""
    from peft import LoraConfig

    return LoraConfig(
        r=lora.r,
        lora_alpha=alpha,
        target_modules=list(lora.target_modules),
        layers_to_transform=list(lora.layers_to_transform) if lora.layers_to_transform else None,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
