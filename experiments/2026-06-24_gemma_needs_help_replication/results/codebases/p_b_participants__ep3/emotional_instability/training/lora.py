"""LoRA configuration (paper §4, Appendix E).

The paper uses "LoRA rank-64 adapters on all layers" for both SFT and DPO. We
build a PEFT ``LoraConfig`` accordingly, and additionally support the §4.2
"internal vs expressed" layer-range ablation, where adapters are restricted to a
contiguous band of decoder layers (e.g. layers 30-35 only, or layer 40 onwards)
to test which layers the intervention must act on.

Unspecified-in-paper choices (documented in DESIGN.md §"LoRA"):
  * alpha = 2*rank (=128), the common PEFT default;
  * dropout = 0.0;
  * "all layers" → all linear projections (attention q/k/v/o + MLP gate/up/down),
    PEFT's ``target_modules="all-linear"`` equivalent expressed explicitly for
    Gemma so the layer-range filter can be applied.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Linear projection names in Gemma-3 decoder blocks.
GEMMA_LINEAR_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",       # attention
    "gate_proj", "up_proj", "down_proj",           # MLP
]


def _target_modules(layer_range: tuple[int, int | None] | None):
    """Resolve target_modules for PEFT.

    With no layer_range we target all linear modules ("all-linear"). With a
    layer_range [start, end] we emit fully-qualified module names restricted to
    ``model.layers.{i}.*`` for i in the (inclusive start, exclusive-or-open end)
    band, which is how PEFT scopes adapters to specific decoder layers.
    """
    if layer_range is None:
        return "all-linear"
    start, end = layer_range
    end = end if end is not None else 1000  # open-ended upper bound
    names = []
    for layer in range(start, end):
        for proj in GEMMA_LINEAR_MODULES:
            names.append(f"model.layers.{layer}.self_attn.{proj}"
                         if proj.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
                         else f"model.layers.{layer}.mlp.{proj}")
    return names


def build_lora_config(
    *,
    rank: int = 64,
    alpha: int = 128,
    dropout: float = 0.0,
    layer_range: tuple[int, int | None] | None = None,
):
    """Return a PEFT LoraConfig for the given rank / layer scope."""
    from peft import LoraConfig

    target_modules = _target_modules(layer_range)
    if layer_range is not None:
        logger.info(
            "LoRA restricted to decoder layers %s (%d target modules) — §4.2 ablation.",
            layer_range, len(target_modules),
        )
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
