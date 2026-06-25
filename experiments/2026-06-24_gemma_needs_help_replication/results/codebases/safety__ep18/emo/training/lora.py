"""LoRA target-module helpers (paper Appendix E + the layer-ablation in App. I).

Adapters are applied to all attention and MLP projections
(q/k/v/o/gate/up/down). For the internal-emotion ablation we also support
restricting adapters to a contiguous range of decoder layers.
"""

from __future__ import annotations

PROJECTIONS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def lora_config(rank: int, alpha: int, layer_range: tuple[int, int] | None = None):
    """Build a PEFT ``LoraConfig``.

    If ``layer_range=(lo, hi)`` is given, adapters are restricted to decoder
    layers ``lo..hi-1`` (suffix-matched module names), reproducing the
    layer-subset DPO runs in Appendix I.
    """
    from peft import LoraConfig

    if layer_range is None:
        target_modules = PROJECTIONS
    else:
        lo, hi = layer_range
        attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
        mlp = ["gate_proj", "up_proj", "down_proj"]
        target_modules = []
        for i in range(lo, hi):
            target_modules += [f"layers.{i}.self_attn.{p}" for p in attn]
            target_modules += [f"layers.{i}.mlp.{p}" for p in mlp]
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
