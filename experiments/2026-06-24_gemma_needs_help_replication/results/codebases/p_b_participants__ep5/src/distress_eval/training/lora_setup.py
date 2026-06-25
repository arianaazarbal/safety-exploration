"""Shared LoRA / model-loading setup for finetuning (Appendix E, Table 9).

Supports restricting LoRA adapters to a contiguous band of decoder layers via
``layer_subset=[start, end)`` — used for the Appendix I layer-ablation showing
the DPO intervention must act on early/central layers (not just the final ones)
to suppress *internal* and not merely expressed emotion."""
from __future__ import annotations

from typing import Any


def build_lora_config(lora_cfg: dict[str, Any], lora_alpha: int):
    from peft import LoraConfig

    kwargs = dict(
        r=lora_cfg["rank"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    subset = lora_cfg.get("layer_subset")
    if subset:
        start, end = subset
        kwargs["layers_to_transform"] = list(range(start, end))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def load_base_for_training(model_id: str, dtype: str = "bfloat16",
                           load_in_4bit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, dtype),
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto",
        attn_implementation="eager", **quant,
    )
    return model, tok
