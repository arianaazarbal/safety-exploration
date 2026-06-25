"""LoRA adapter configuration (rank-64, all layers) + the layer-range ablation.

Section 4.2 ("internal vs expressed emotions") needs layer-restricted variants:
adapters on layers 30-35 only are nearly as effective as all layers, whereas
adapters from layer 40 onwards are not. ``LoRAConfig.layer_range`` drives this via
PEFT's ``layers_to_transform``.
"""

from __future__ import annotations

from ..config import LoRAConfig


def build_peft_config(lora: LoRAConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if lora.layer_range is not None:
        lo, hi = lora.layer_range
        kwargs["layers_to_transform"] = list(range(lo, hi + 1))
        # Gemma decoder layers live under model.layers.<i>.
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
