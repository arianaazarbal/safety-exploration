"""Shared LoRA configuration (Appendix E / Table 9).

LoRA adapters target all attention + MLP projections. The optional ``layer_range``
restricts adapters to an inclusive band of decoder layers, used by the Appendix I
layer-ablation experiment.
"""

from __future__ import annotations

from ..config import LORA_TARGET_MODULES


def build_lora_config(rank: int, alpha: int, layer_range: tuple[int, int] | None = None):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )
    if layer_range is not None:
        lo, hi = layer_range
        # inclusive band; peft matches modules under ".layers.<i>."
        kwargs["layers_to_transform"] = list(range(lo, hi + 1))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
