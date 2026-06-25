"""Factory that turns a model handle into a live `ModelInterface`.

Handles the registered target models from `config.TARGET_MODELS`, plus the
fine-tuned Gemma variants from Section 4, which are expressed as the base
instruct model + a LoRA adapter directory:

    build_model("gemma-3-27b-it")                         # vanilla instruct
    build_model("gemma-3-27b-it", adapter_dir="...dpo")   # DPO finetune
    build_model("gemma-3-27b-it-dpo")                      # shorthand for the above
"""

from __future__ import annotations

from typing import Optional

import config

from .base import ModelInterface


# Convenience shorthands for the Section 4 fine-tunes.
_FINETUNE_SHORTHANDS = {
    "gemma-3-27b-it-dpo": ("gemma-3-27b-it", config.DPO_ADAPTER_DIR),
    "gemma-3-27b-it-sft-diverse": ("gemma-3-27b-it", config.SFT_DIVERSE_ADAPTER_DIR),
    "gemma-3-27b-it-sft-teacher": ("gemma-3-27b-it", config.SFT_TEACHER_ADAPTER_DIR),
}


def build_model(name: str, *, adapter_dir: Optional[str] = None, **kwargs) -> ModelInterface:
    if name in _FINETUNE_SHORTHANDS:
        base_name, default_adapter = _FINETUNE_SHORTHANDS[name]
        spec = config.TARGET_MODELS[base_name]
        adapter_dir = str(adapter_dir or default_adapter)
        from .hf_gemma import HFGemmaModel

        return HFGemmaModel(spec, adapter_dir=adapter_dir, **kwargs)

    if name not in config.TARGET_MODELS:
        raise KeyError(f"Unknown model handle {name!r}. "
                       f"Known: {sorted(config.TARGET_MODELS)} + "
                       f"{sorted(_FINETUNE_SHORTHANDS)}")

    spec = config.TARGET_MODELS[name]
    if spec.backend == "hf":
        from .hf_gemma import HFGemmaModel

        return HFGemmaModel(spec, adapter_dir=adapter_dir, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter import OpenRouterModel

        return OpenRouterModel(spec)
    raise ValueError(f"Unknown backend {spec.backend!r} for model {name!r}")
