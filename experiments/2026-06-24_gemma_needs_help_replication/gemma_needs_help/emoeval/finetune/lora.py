"""LoRA config builder shared by the SFT and DPO trainers.

Supports the Section 4.2 layer-restriction ablation: by default LoRA adapters
target ALL layers; setting `config.FINETUNE.lora_layer_window = (lo, hi)`
restricts them to a contiguous layer window (the paper finds layers 30-35 alone
are nearly as effective as all layers, whereas layer 40+ is not).
"""
from __future__ import annotations

import re

from .. import config


def _layer_index(module_name: str):
    m = re.search(r"layers\.(\d+)\.", module_name)
    return int(m.group(1)) if m else None


def build_lora_config(model):
    from peft import LoraConfig

    fc = config.FINETUNE
    window = fc.lora_layer_window

    if window is None:
        target_modules = list(fc.lora_target_modules)
    else:
        lo, hi = window
        # Enumerate concrete module paths inside the window.
        target_modules = []
        for name, _ in model.named_modules():
            idx = _layer_index(name)
            if idx is None:
                continue
            if idx < lo or (hi is not None and idx > hi):
                continue
            if any(name.endswith(t) for t in fc.lora_target_modules):
                target_modules.append(name)

    return LoraConfig(
        r=fc.lora_rank,
        lora_alpha=fc.lora_alpha,
        lora_dropout=fc.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
