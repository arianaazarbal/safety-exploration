"""Build layer-restricted LoRA target-module lists (Appendix I ablations).

PEFT's `target_modules` accepts fully-qualified module name suffixes. For Gemma 3
the decoder layers are `model.layers.{i}.<proj>`, so to restrict adapters to a
subset of layers we enumerate the exact module paths for those layers only.
"""
from __future__ import annotations

import config


def lora_target_modules_for_layers(num_layers: int, spec) -> list[str]:
    """Return explicit `model.layers.{i}.{proj}` names for the requested layers.

    ``spec`` is one of:
      * None            -> all layers (returns the bare proj names, PEFT matches all)
      * "last:N"        -> the final N layers
      * (start, end)    -> inclusive-exclusive layer range, clipped to model size
    """
    if spec is None:
        return list(config.LORA_TARGET_MODULES)

    if isinstance(spec, str) and spec.startswith("last:"):
        n = int(spec.split(":", 1)[1])
        layers = range(max(0, num_layers - n), num_layers)
    elif isinstance(spec, tuple):
        start, end = spec
        layers = range(max(0, start), min(num_layers, end))
    else:
        raise ValueError(f"bad layer spec: {spec!r}")

    modules = []
    for i in layers:
        for proj in config.LORA_TARGET_MODULES:
            modules.append(f"model.layers.{i}.{('self_attn' if 'proj' in proj and proj in {'q_proj','k_proj','v_proj','o_proj'} else 'mlp')}.{proj}")
    return modules
