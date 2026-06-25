"""LoRA target-module helpers.

By default LoRA is applied to all attention + MLP projections on every layer
(Appendix E). For the Appendix I layer-subset ablation, ``target_modules_for_layers``
restricts adapters to an explicit layer range (e.g. layers 30-35 only).
"""
from __future__ import annotations

PROJ_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def target_modules_for_layers(model, layer_range: tuple[int, int] | None):
    """Return a target_modules list for PEFT.

    ``layer_range`` is an inclusive-exclusive (start, end) over decoder layer
    indices. ``None`` -> all layers (PEFT receives the bare module names).
    """
    if layer_range is None:
        return PROJ_MODULES
    start, end = layer_range
    # Discover fully-qualified names so we can pin specific layer indices.
    names = []
    for full_name, _ in model.named_modules():
        for proj in PROJ_MODULES:
            if full_name.endswith(proj) and ".layers." in full_name:
                idx = int(full_name.split(".layers.")[1].split(".")[0])
                if start <= idx < end:
                    names.append(full_name)
    if not names:
        raise ValueError(f"No projection modules found for layers {layer_range}")
    return names


def parse_layer_range(s: str | None) -> tuple[int, int] | None:
    if not s:
        return None
    a, b = s.split("-")
    return int(a), int(b)
