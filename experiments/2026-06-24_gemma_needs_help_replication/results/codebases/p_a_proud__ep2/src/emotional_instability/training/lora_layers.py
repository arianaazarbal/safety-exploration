"""LoRA config construction, including the layer-restriction used by the App. I ablations.

PEFT can restrict adapters to a subset of decoder layers via ``layers_to_transform`` +
``layers_pattern``. The App. I experiments train DPO with adapters on only certain layer
ranges (e.g. 30-35) to show the intervention must act on central/early layers, not just the
final ones. ``layer_range=(lo, hi)`` means layers ``lo..hi-1`` (half-open, matching the
config sweep tuples).
"""
from __future__ import annotations

from ..config import LoRAConfig


def target_modules_for_layers(layer_range: tuple[int, int] | None) -> list[int] | None:
    """Return the explicit layer indices to adapt, or None for all layers."""
    if layer_range is None:
        return None
    lo, hi = layer_range
    return list(range(lo, hi))


def build_lora_config(cfg: LoRAConfig):
    """Build a peft.LoraConfig from our LoRAConfig (imported lazily to keep peft optional)."""
    from peft import LoraConfig

    layers = target_modules_for_layers(cfg.layer_range)
    kwargs = dict(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers is not None:
        # Restrict adapters to the given decoder layers (Gemma layers are under `...layers.N`).
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
