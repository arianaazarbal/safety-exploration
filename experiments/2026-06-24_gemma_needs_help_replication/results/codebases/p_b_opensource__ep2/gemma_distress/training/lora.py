"""PEFT LoRA configuration, including the Appendix-I layer-subset ablations.

The paper applies rank-64 LoRA to "all attention and MLP projection layers"
(q/k/v/o_proj, gate/up/down_proj) on every layer (Table 9 / App E). Appendix I
re-runs DPO with adapters restricted to layer subsets (e.g. 30–35, last-20,
40–50) to show the intervention must touch central layers, not only the final
ones. ``build_lora_config`` exposes that via ``layers`` (an inclusive
``(start, end)`` range or explicit list).
"""

from __future__ import annotations

from typing import Optional, Union

from .. import config

LayerSpec = Union[None, tuple[int, int], list[int]]


def _resolve_layers(layers: LayerSpec) -> Optional[list[int]]:
    if layers is None:
        return None
    if isinstance(layers, tuple) and len(layers) == 2:
        start, end = layers
        return list(range(start, end + 1))  # inclusive, matching App I "layers 30-35"
    return list(layers)


def build_lora_config(
    *,
    r: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
    target_modules: Optional[tuple] = None,
    layers: LayerSpec = None,
):
    """Return a PEFT ``LoraConfig``. ``layers`` restricts adapters to a subset of
    decoder layers (Appendix I); ``None`` means all layers."""
    from peft import LoraConfig

    target_modules = list(target_modules or config.LoRAConfig.target_modules)
    layers_to_transform = _resolve_layers(layers)
    kwargs = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers_to_transform is not None:
        kwargs["layers_to_transform"] = layers_to_transform
    return LoraConfig(**kwargs)


# Appendix-I ablation grid (inclusive layer ranges). Used by the layer-ablation
# CLI to reproduce Figures 12–13.
APPENDIX_I_LAYER_SETS = {
    "all": None,
    "last5": (43, 47),     # Gemma-3-27B has 48 decoder layers (0-47); "final 5"
    "last20": (28, 47),
    "last30": (18, 47),
    "l20_25": (20, 25),
    "l25_30": (25, 30),
    "l30_35": (30, 35),
    "l35_40": (35, 40),
    "l40_50": (40, 47),
}
