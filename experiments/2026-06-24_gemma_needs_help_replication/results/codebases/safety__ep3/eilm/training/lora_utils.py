"""Shared LoRA config builder (Appendix E + Appendix I layer ablations).

Adapters are applied to all attention + MLP projections by default. For the
layer-localisation ablation (Appendix I) a subset of decoder layers can be
specified via ``layer_subset`` (e.g. ``range(30, 36)`` for "layers 30-35 only"),
which the paper found nearly as effective as adapting all layers — evidence the
intervention acts on internal emotional representations, not just the output
head.
"""

from __future__ import annotations

from collections.abc import Iterable

from .. import config


def build_lora_config(
    rank: int,
    alpha: int,
    layer_subset: Iterable[int] | None = None,
):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.TRAIN.lora_target_modules),
    )
    if layer_subset is not None:
        # Restrict adapters to specific decoder layers.
        kwargs["layers_to_transform"] = list(layer_subset)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
