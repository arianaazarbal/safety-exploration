"""Shared finetuning utilities: LoRA config (rank-64, all proj layers) and
dataset loading. Supports the Appendix I layer-subset ablation via
`layers_to_transform`."""
from __future__ import annotations

import config

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def make_lora_config(rank: int, alpha: int, layers: list[int] | None = None):
    """Build a PEFT LoraConfig. If `layers` is given, adapters are applied only
    to those decoder layer indices (Appendix I: e.g. layers 30-35)."""
    from peft import LoraConfig

    kwargs = dict(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )
    if layers is not None:
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def load_jsonl_dataset(path: str):
    from datasets import load_dataset

    return load_dataset("json", data_files=str(path), split="train")


def parse_layers(spec: str | None) -> list[int] | None:
    """Parse a layer spec like "30-35" or "0,1,2" into a list of indices."""
    if not spec:
        return None
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b)))
    return [int(x) for x in spec.split(",")]


def load_base_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = config.ALL_MODELS[config.FINETUNE_BASE].model_id
    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    return model, tok, base_id
