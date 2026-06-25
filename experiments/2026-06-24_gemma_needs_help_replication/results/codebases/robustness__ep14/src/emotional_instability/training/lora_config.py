"""LoRA configuration helper (Appendix E / Table 9).

rank-64 adapters on all attention + MLP projections by default. Optionally restrict
to a subset of decoder layers (Appendix I layer ablations) via `layers`.
"""
from __future__ import annotations

from typing import Sequence


def build_lora_config(
    rank: int = 64,
    alpha: int = 64,
    target_modules: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    dropout: float = 0.0,
):
    from peft import LoraConfig

    target_modules = list(target_modules or
                          ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"])
    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers is not None:
        # restrict adapters to specific decoder layer indices (layer-subset ablation)
        kwargs["layers_to_transform"] = list(layers)
    return LoraConfig(**kwargs)
