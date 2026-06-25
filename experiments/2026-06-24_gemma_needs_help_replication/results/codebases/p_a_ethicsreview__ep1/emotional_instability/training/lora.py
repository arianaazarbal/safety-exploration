"""LoRA configuration, including the layer-restricted ablations (Section 4.2).

The default adapter targets all linear layers (paper: "rank-64 adapters on all
layers"). The internal-vs-expressed-emotion ablation restricts the adapter to a
subset of decoder layers:
  * ``layers_30_35`` - adapters on layers 30-35 only (paper: nearly as effective
    as all layers), and
  * ``layers_40_plus`` - adapters from layer 40 onward (paper: does NOT
    effectively reduce distress),
which together argue the intervention must act on early/central layers, i.e. it
suppresses internal as well as externalised emotion.
"""

from __future__ import annotations

from typing import Any


def build_lora_config(lora_cfg: dict[str, Any], ablation: str = "all"):
    """Return a peft ``LoraConfig`` for the given ablation key.

    ``ablation`` indexes ``lora_cfg['layer_ablations']``; ``"all"`` (or a None
    entry) means no layer restriction.
    """
    from peft import LoraConfig

    layers = lora_cfg.get("layer_ablations", {}).get(ablation)
    kwargs: dict[str, Any] = dict(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    if layers:  # restrict to a specific set of decoder layers
        kwargs["layers_to_transform"] = list(layers)
    return LoraConfig(**kwargs)
