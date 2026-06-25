"""LoRA configuration builder (Appendix E, Table 9).

Rank-64 adapters on all attention and MLP projection layers. Supports
restricting the adapter to a subset of decoder layers, which is what the
layer-ablation experiments (Appendix I) require.
"""
from __future__ import annotations

from typing import Optional

from .. import config


def build_lora_config(
    lora: config.LoRAConfig,
    layer_range: Optional[tuple] = None,
):
    """Return a ``peft.LoraConfig``.

    If ``layer_range=(lo, hi)`` is given, the adapter is applied only to decoder
    layers in ``range(lo, hi)`` via ``layers_to_transform`` (Appendix I ablations).
    """
    from peft import LoraConfig

    kwargs = dict(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layer_range is not None:
        lo, hi = layer_range
        kwargs["layers_to_transform"] = list(range(lo, hi))
    return LoraConfig(**kwargs)
