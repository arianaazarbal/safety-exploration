"""LoRA configuration helpers (Section 4.1 + the layer-ablation of Section 4.2).

Both SFT and DPO use rank-64 LoRA adapters on all layers. Section 4.2's
internal-vs-expressed analysis additionally restricts adapters to a layer band
(e.g. layers 30-35 only, or layer 40+), which ``target_layers`` supports.
"""
from __future__ import annotations

from typing import Optional

# attention + MLP projection module names for Gemma-3 decoder layers
_GEMMA_TARGET_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


def build_lora_config(rank: int = 64, alpha: int = 128, dropout: float = 0.05,
                      target_layers: Optional[list[int]] = None):
    """Return a peft LoraConfig.

    If ``target_layers`` is given, only those decoder layer indices receive
    adapters (used for the layer-ablation in Section 4.2); otherwise all layers
    are adapted.
    """
    from peft import LoraConfig

    if target_layers is None:
        target_modules = _GEMMA_TARGET_SUFFIXES
        layers_to_transform = None
    else:
        # peft matches `target_modules` by suffix and restricts to the given
        # layer indices via `layers_to_transform` + `layers_pattern`.
        target_modules = _GEMMA_TARGET_SUFFIXES
        layers_to_transform = list(target_layers)

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        layers_pattern="layers" if target_layers is not None else None,
    )
