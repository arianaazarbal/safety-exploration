"""Shared LoRA configuration (Appendix E, Table 9 / Appendix I).

LoRA adapters on all attention and MLP projections by default. `layers` lets
the Appendix I ablation restrict adapters to a subset of decoder layers
(e.g. [30, 31, 32, 33, 34] for the "layers 30-35 only" run).
"""
from __future__ import annotations

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def make_lora_config(rank: int, alpha: int, layers: list[int] | None = None):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    if layers is not None:
        # Restrict adapters to specific decoder layers (Appendix I ablation).
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
