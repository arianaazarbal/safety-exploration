"""LoRA configuration (Appendix E, Table 9; Appendix I layer ablations).

Rank-64 adapters on all attention + MLP projection layers. ``layers`` (when set)
restricts the adapters to a subset of transformer blocks via PEFT's
``layers_to_transform`` — this is the knob the Appendix I ablation uses to show
that adapters on early/central layers (e.g. 30–35) are necessary to suppress
distress, while layer-40+ adapters are largely ineffective.
"""

from __future__ import annotations

from ..config import LoRAConfig


def build_peft_config(lora: LoRAConfig, *, alpha: int):
    from peft import LoraConfig, TaskType

    kwargs = dict(
        r=lora.rank,
        lora_alpha=alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    if lora.layers is not None:
        # Restrict adapters to a subset of decoder layers (Appendix I).
        kwargs["layers_to_transform"] = list(lora.layers)
    return LoraConfig(**kwargs)
