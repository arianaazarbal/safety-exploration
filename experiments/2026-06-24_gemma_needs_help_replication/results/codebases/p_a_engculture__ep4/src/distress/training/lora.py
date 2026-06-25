"""Build a PEFT ``LoraConfig`` from our :class:`distress.config.LoRAConfig`,
including the optional contiguous layer-range restriction used by the Appendix I
ablation (e.g. adapters on layers 30-35 only)."""

from __future__ import annotations

from ..config import LoRAConfig


def build_lora_config(cfg: LoRAConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.layers_start is not None or cfg.layers_end is not None:
        start = cfg.layers_start or 0
        end = cfg.layers_end  # exclusive; None means "to the end"
        # layers_to_transform restricts adapters to specific decoder-layer indices.
        layers = list(range(start, end)) if end is not None else None
        if layers is not None:
            kwargs["layers_to_transform"] = layers
            kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
