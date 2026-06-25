"""LoRA configuration, including the layer-subset ablation (Appendix I).

The internal-emotion ablation re-runs DPO with adapters restricted to a subset
of transformer layers. PEFT supports this via ``layers_to_transform`` (a list of
layer indices) together with ``layers_pattern`` (the attribute name of the layer
list in the model). For Gemma the decoder layers live under
``model.layers``, so ``layers_pattern="layers"``.
"""

from __future__ import annotations


def _layer_indices(layers, num_layers: int = 50) -> list[int] | None:
    """Resolve a config ``layers`` value to explicit indices, or None for 'all'."""
    if layers in (None, "all"):
        return None
    if isinstance(layers, (list, tuple)) and len(layers) == 2:
        start, end = int(layers[0]), int(layers[1])
        return list(range(start, min(end, num_layers)))
    if isinstance(layers, (list, tuple)):
        return [int(x) for x in layers]
    raise ValueError(f"Unrecognised layers spec: {layers!r}")


def lora_config(
    r: int = 64,
    alpha: int = 64,
    target_modules=None,
    layers="all",
    num_layers: int = 50,
    dropout: float = 0.0,
):
    """Build a PEFT ``LoraConfig`` (optionally restricted to a layer range)."""
    from peft import LoraConfig

    target_modules = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]
    layers_to_transform = _layer_indices(layers, num_layers)
    kwargs = dict(
        r=r,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers_to_transform is not None:
        kwargs["layers_to_transform"] = layers_to_transform
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
