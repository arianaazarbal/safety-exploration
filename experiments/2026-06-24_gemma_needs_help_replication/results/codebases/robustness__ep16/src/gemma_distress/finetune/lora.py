"""LoRA config builder shared by the DPO and SFT trainers (Table 9, Appendix I).

Target modules are all attention + MLP projections. ``lora_layers`` optionally
restricts adapters to a subset of decoder layers (the Appendix I ablation that
shows central layers 25-35 carry the intervention); ``None`` => all layers.
"""

from __future__ import annotations


def build_lora_config(
    rank: int,
    alpha: int,
    target_modules: list[str],
    lora_layers: list[int] | None = None,
    dropout: float = 0.0,
):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if lora_layers is not None:
        # Restrict adapters to specific decoder layers (Appendix I ablation).
        kwargs["layers_to_transform"] = list(lora_layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
