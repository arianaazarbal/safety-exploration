"""LoRA configuration helper (Appendix E, Table 9, and the Appendix I layer
ablations)."""
from __future__ import annotations

from ..config import LoraConfig as LoraCfg


def build_lora_config(cfg: LoraCfg):
    """Build a PEFT ``LoraConfig`` from our dataclass.

    ``layers_to_transform`` restricts the adapter to a subset of decoder layers
    for the Appendix I ablations (e.g. layers 30-35 only); ``None`` applies LoRA
    to all layers, matching the main DPO/SFT runs.
    """
    from peft import LoraConfig as PeftLoraConfig

    kwargs = dict(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(cfg.layers_to_transform)
    return PeftLoraConfig(**kwargs)
