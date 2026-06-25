"""LoRA configuration shared by DPO and SFT (Appendix E + Appendix I ablation).

Default: rank-64 adapters on all attention + MLP projection layers. The
``layers_to_transform`` band reproduces the Appendix I ablation (e.g. layers
30-35 only, or >=40) used to show the intervention must act on early/central
layers to suppress *internal* — not just expressed — emotion.
"""
from __future__ import annotations

from emoinstab.config import LoRAConfig


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
    if cfg.layers_to_transform is not None:
        lo, hi = cfg.layers_to_transform
        kwargs["layers_to_transform"] = list(range(lo, hi + 1))
    return LoraConfig(**kwargs)
