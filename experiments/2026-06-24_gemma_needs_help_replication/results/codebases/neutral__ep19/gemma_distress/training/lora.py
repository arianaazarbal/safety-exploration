"""Build PEFT LoRA configs, including the layer-band restriction used by the
App. I ablations (restricting adapters to a contiguous range of decoder layers)."""
from __future__ import annotations

from .. import config_shim as cfg


def build_peft_config(*, rank, alpha, target_modules, layer_band=None, task_type="CAUSAL_LM"):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(target_modules),
        task_type=task_type,
    )
    if layer_band is not None:
        lo, hi = layer_band
        kwargs["layers_to_transform"] = list(range(lo, hi))
        # Gemma decoder layers live under model.layers.<idx>.
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
