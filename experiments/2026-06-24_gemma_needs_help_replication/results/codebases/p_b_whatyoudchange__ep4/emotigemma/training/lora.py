"""Shared LoRA configuration, including the layer-range ablations (Section 4.2).

The paper uses rank-64 LoRA adapters on all layers, and ablates the layer range
to show the intervention must act on early layers (layers 30-35 alone ≈ all
layers; layer 40+ alone is ineffective).
"""
from __future__ import annotations


def parse_layer_spec(spec: str, n_layers: int) -> list[int] | None:
    """Translate a layer-ablation spec into a list of layer indices, or None=all.

    "all"   -> None (every layer)
    "30-35" -> [30, 31, 32, 33, 34, 35]
    "40+"   -> [40, 41, ..., n_layers-1]
    """
    if spec == "all":
        return None
    if spec.endswith("+"):
        start = int(spec[:-1])
        return list(range(start, n_layers))
    lo, hi = spec.split("-")
    return list(range(int(lo), int(hi) + 1))


def build_lora_config(rank: int, target: str, layers: list[int] | None):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if target == "all-linear":
        kwargs["target_modules"] = "all-linear"
    else:
        kwargs["target_modules"] = target.split(",")
    if layers is not None:
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
