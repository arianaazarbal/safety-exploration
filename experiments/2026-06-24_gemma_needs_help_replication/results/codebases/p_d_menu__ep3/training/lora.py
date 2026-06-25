"""Shared LoRA configuration (Section 4.1 / Appendix E).

Paper: "LoRA rank-64 adapters on all layers." Section 4.2's internal-vs-expressed
ablation restricts adapters to specific layer ranges (e.g. layers 30-35 only, or
from layer 40 onwards), which we expose via `target_layers`.
"""

from __future__ import annotations


def build_lora_config(rank: int = 64, alpha: int | None = None,
                      target_layers: list[int] | None = None):
    """Return a peft.LoraConfig.

    target_layers=None -> adapters on all layers (default). A list restricts the
    adapter to those decoder-layer indices (the Section 4.2 ablation).
    """
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha or rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # "all-linear" attaches to every linear projection (attn + MLP), matching
        # "adapters on all layers".
        target_modules="all-linear",
    )
    if target_layers is not None:
        kwargs["layers_to_transform"] = list(target_layers)
    return LoraConfig(**kwargs)


# Convenience presets for the Section 4.2 ablation.
LAYERS_30_35 = list(range(30, 36))     # "layers 30-35 only" — nearly as effective
LAYERS_FROM_40 = None                  # set at call time once n_layers is known


def layers_from(start: int, n_layers: int) -> list[int]:
    return list(range(start, n_layers))
