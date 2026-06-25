"""LoRA configuration (Appendix E, Table 9) + layer-subset ablation (Appendix I).

All LoRA runs target the attention and MLP projections:
    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

Hyperparameters (Table 9):
    DPO: rank 64, alpha 64
    SFT: rank 64, alpha 128

The Appendix-I ablation restricts adapters to a contiguous layer range
(e.g. layers 30-35) via PEFT's `layers_to_transform`. The finding is that
layers <40 are necessary; 30-35 alone is nearly as effective as all layers,
while 40-50 is largely ineffective — evidence the intervention acts on internal
states, not just final-layer expression.
"""

from __future__ import annotations

from typing import Optional, Sequence

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def make_lora_config(rank: int = 64, alpha: int = 64,
                     layers: Optional[Sequence[int]] = None,
                     dropout: float = 0.0):
    """Build a PEFT LoraConfig. `layers` (if given) restricts adapters to those
    decoder-layer indices (Appendix-I ablation)."""
    from peft import LoraConfig
    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers is not None:
        kwargs["layers_to_transform"] = list(layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


# Named ablation configs from Appendix I / Figures 12-13.
ABLATION_LAYER_RANGES = {
    "all": None,
    "layers_20_25": range(20, 25),
    "layers_25_30": range(25, 30),
    "layers_30_35": range(30, 35),   # ~as effective as all
    "layers_35_40": range(35, 40),
    "layers_40_50": range(40, 50),   # largely ineffective
    "last_5": range(-5, 0),          # final 5 layers only (insufficient)
}
