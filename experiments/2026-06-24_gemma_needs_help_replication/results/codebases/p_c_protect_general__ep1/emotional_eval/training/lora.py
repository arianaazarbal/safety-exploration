"""Shared LoRA configuration (Appendix E, Table 9).

Adapters are applied to all attention and MLP projection layers:
``q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`` at rank 64.
``layers_to_transform`` exposes the layer-subset ablations from Appendix I
(e.g. layers 30--35 only, or layers 40+).
"""

from __future__ import annotations

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_lora_config(
    *,
    rank: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
    layers_to_transform: list[int] | None = None,
):
    """Return a PEFT ``LoraConfig``.

    ``layers_to_transform`` restricts adapters to specific decoder layer
    indices, used by the internal-vs-expressed-emotion analysis (Appendix I):
    adapters on layers 30--35 are nearly as effective as all layers, while
    adapters from layer 40 onward do not reduce distress.
    """
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        layers_to_transform=layers_to_transform,
    )
