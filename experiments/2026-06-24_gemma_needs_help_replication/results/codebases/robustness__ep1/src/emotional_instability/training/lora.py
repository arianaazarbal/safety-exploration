"""Shared LoRA config construction, including the Appendix I layer-subset option."""
from __future__ import annotations

from typing import Optional


def parse_layers(spec) -> Optional[list[int]]:
    """Parse ``lora_layers`` config: "all" -> None (all layers), or "30-35"/"30-40"
    -> the inclusive-exclusive range used by the Appendix I ablation, or a list.
    """
    if spec in (None, "all"):
        return None
    if isinstance(spec, list):
        return [int(x) for x in spec]
    if isinstance(spec, str) and "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi)))
    return [int(spec)]


def build_lora_config(cfg, method: str):
    """Build a PEFT LoraConfig for 'sft' or 'dpo' from config.yaml."""
    from peft import LoraConfig

    tcfg = cfg["training"]
    mcfg = tcfg[method]
    layers = parse_layers(tcfg.get("lora_layers", "all"))
    kwargs = dict(
        r=mcfg["lora_rank"],
        lora_alpha=mcfg["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=tcfg["lora_target_modules"],
    )
    if layers is not None:
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
