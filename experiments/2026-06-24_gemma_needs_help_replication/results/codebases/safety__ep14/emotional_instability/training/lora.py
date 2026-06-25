"""LoRA config helpers shared by the SFT and DPO trainers.

Supports the layer-subset ablation from Appendix I via `layers_to_transform`,
so we can train adapters on only e.g. layers 30-35 to localise the intervention.
"""
from __future__ import annotations


def resolve_layer_range(layer_range, n_layers: int) -> list[int] | None:
    """Turn a [start, end] spec (with Python-style negatives / None) into an
    explicit list of layer indices, or None to mean 'all layers'."""
    if layer_range is None:
        return None
    start, end = layer_range
    if start is None:
        start = 0
    if end is None:
        end = n_layers
    if start < 0:
        start = n_layers + start
    if end < 0:
        end = n_layers + end
    start = max(0, min(start, n_layers))
    end = max(0, min(end, n_layers))
    return list(range(start, end))


def make_lora_config(
    rank: int,
    alpha: int,
    target_modules: list[str],
    layers_to_transform: list[int] | None = None,
    dropout: float = 0.0,
):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
    )


def count_layers(model) -> int:
    """Best-effort decoder-layer count for Gemma-style models."""
    for attr in ("model",):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "layers"):
            return len(base.layers)
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    raise RuntimeError("Could not determine number of layers for LoRA layer ablation.")
