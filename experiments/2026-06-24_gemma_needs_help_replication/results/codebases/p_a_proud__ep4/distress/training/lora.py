"""LoRA configuration (Paper Appendix E, Table 9; Appendix I layer ablations).

Rank-64 adapters on all attention + MLP projections by default. ``layer_range``
optionally restricts adapters to a contiguous band of decoder layers, which is
how Appendix I tests whether the DPO intervention must act on early/central
layers (e.g. ``[30, 35]`` => layers 30..34 only).
"""

from __future__ import annotations


def build_lora_config(
    *,
    rank: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    layer_range: list[int] | None = None,
):
    from peft import LoraConfig

    target_modules = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    kwargs: dict = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layer_range is not None:
        start, end = layer_range
        kwargs["layers_to_transform"] = list(range(start, end))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
