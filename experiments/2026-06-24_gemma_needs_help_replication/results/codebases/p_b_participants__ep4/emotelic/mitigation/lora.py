"""LoRA configuration (Appendix E) + layer-range ablation support (Section 4.2).

Adapters are applied to all attention and MLP projections. The layer ablation
("layers 30-35 only" vs "from layer 40 onwards") is expressed via PEFT's
`layers_to_transform`, which restricts adaptation to specific decoder layers —
this is how the paper localises the intervention to early/central layers.
"""
from __future__ import annotations

ALL_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",     # attention
    "gate_proj", "up_proj", "down_proj",        # MLP
]


def lora_config(
    *,
    rank: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
    layers: list[int] | None = None,   # None == all layers; else restrict (ablation)
):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=ALL_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers is not None:
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


# Named ablation presets used in Section 4.2 ("internal vs expressed emotions").
ABLATION_LAYER_SETS = {
    "all": None,
    "layers_30_35": list(range(30, 36)),     # nearly as effective as all layers
    "layers_40_plus": list(range(40, 62)),   # NOT effective (late layers only)
}
