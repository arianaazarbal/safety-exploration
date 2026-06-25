"""LoRA configuration, including the Appendix I layer-subset ablation.

``config/training.yaml`` ``lora.layers`` is either ``"all"`` (main result) or a
``[start, end)`` range. The range maps to PEFT's ``layers_to_transform`` so that
adapters are applied only to decoder layers in ``range(start, end)`` -- this is
how Appendix I shows that early/central layers (e.g. 30-35) carry the effect while
layers >= 40 do not.
"""

from __future__ import annotations

from typing import Optional


def build_lora_config(training_cfg: dict, method: str):
    """Return a ``peft.LoraConfig`` for the given method ('dpo' or 'sft')."""
    from peft import LoraConfig

    method_cfg = training_cfg[method]
    lora_cfg = training_cfg["lora"]

    layers_to_transform: Optional[list[int]] = None
    layers_pattern: Optional[str] = None
    layers = lora_cfg.get("layers", "all")
    if layers != "all":
        start, end = layers
        layers_to_transform = list(range(start, end))
        layers_pattern = "layers"   # Gemma decoder stack: model.layers.{i}

    return LoraConfig(
        r=method_cfg["lora_rank"],
        lora_alpha=method_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
