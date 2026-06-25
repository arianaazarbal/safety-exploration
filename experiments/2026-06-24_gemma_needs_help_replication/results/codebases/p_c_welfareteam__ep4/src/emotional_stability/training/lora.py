"""LoRA target-module construction (Section 4.1 / Appendix E & I).

Adapters are applied to all attention and MLP projections (q/k/v/o_proj,
gate/up/down_proj). For the Appendix I layer-localisation study, adapters can be
restricted to a contiguous range of decoder layers (e.g. layers 30-35 only),
which the paper finds is nearly as effective as adapting all layers.
"""

from __future__ import annotations

PROJECTION_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def target_modules(layer_range: tuple[int, int] | None = None) -> list[str]:
    """Return PEFT ``target_modules``.

    With ``layer_range=None`` returns the bare projection names, which PEFT
    matches across all layers. With a ``(start, end)`` range (end exclusive)
    returns fully-qualified module names restricted to those decoder layers,
    matching the Gemma module path ``model.layers.{i}.{...}.{proj}``.
    """
    if layer_range is None:
        return list(PROJECTION_MODULES)
    start, end = layer_range
    names: list[str] = []
    for layer in range(start, end):
        for proj in PROJECTION_MODULES:
            if proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                names.append(f"model.layers.{layer}.self_attn.{proj}")
            else:
                names.append(f"model.layers.{layer}.mlp.{proj}")
    return names
