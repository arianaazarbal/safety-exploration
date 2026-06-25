"""Shared LoRA / model-loading helpers for the finetunes (Appendix E)."""

from __future__ import annotations

from .. import config


def build_lora_config(lora: config.LoRAConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if lora.layers_to_transform is not None:
        # Appendix I layer-subset ablations.
        kwargs["layers_to_transform"] = list(lora.layers_to_transform)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def load_base_model(model_id: str = config.BASE_FINETUNE_MODEL.model_id, load_in_4bit: bool = False):
    """Load Gemma-3-27B-it for finetuning (bf16, optionally 4-bit QLoRA)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation="eager",  # Gemma-3 recommends eager attention
    )
    return model, tokenizer
