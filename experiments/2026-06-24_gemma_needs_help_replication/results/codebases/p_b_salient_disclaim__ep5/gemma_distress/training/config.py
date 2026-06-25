"""Training hyperparameters (Table 9, Appendix E)."""

from __future__ import annotations

from dataclasses import dataclass, field

# LoRA applied to all attention + MLP projections (Appendix E).
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    method: str                 # "dpo" | "sft"
    epochs: int
    learning_rate: float
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    dpo_beta: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: list(TARGET_MODULES))
    # Appendix I: restrict LoRA to a subset of decoder layers [lo, hi). None = all.
    layer_subset: tuple[int, int] | None = None


DPO = TrainConfig(method="dpo", epochs=1, learning_rate=5e-5,
                  lora_rank=64, lora_alpha=64, effective_batch_size=8, dpo_beta=0.1)

# SFT: 2 epochs, lr 1e-4, LoRA rank 64, alpha 128 (Table 9).
SFT = TrainConfig(method="sft", epochs=2, learning_rate=1e-4,
                  lora_rank=64, lora_alpha=128, effective_batch_size=8)
