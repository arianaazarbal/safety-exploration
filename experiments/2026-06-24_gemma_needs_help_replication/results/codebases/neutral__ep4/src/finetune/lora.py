"""LoRA configuration helper, including the layer-subset ablation (Appendix I).

The paper applies rank-64 LoRA adapters to all attention + MLP projection
layers. Appendix I additionally restricts adapters to subsets of layers (e.g.
30-35 only) to show the intervention must act on central layers; we expose this
via `layers`.
"""

from __future__ import annotations


def build_lora_config(rank: int, alpha: int, target_modules, layers=None):
    """Return a peft.LoraConfig. If `layers` is given, only those decoder-layer
    indices receive adapters (via layers_to_transform)."""
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    if layers is not None:
        kwargs["layers_to_transform"] = list(layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
