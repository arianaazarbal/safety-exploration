"""Training hyperparameters (Appendix E, Table 9).

                          DPO            SFT
  Dataset size          280 pairs    1,150 samples
  Epochs                    1             2
  Learning rate           5e-5          1e-4
  LoRA rank                64            64
  LoRA alpha               64           128
  Effective batch size      8            8
  DPO beta                 0.1           —

LoRA is applied to all attention + MLP projections (q/k/v/o, gate/up/down).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DPOHParams:
    n_pairs: int = 280
    rejected_min_score: int = 3
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    beta: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )
    # Appendix I: restrict LoRA to these layer indices (None = all layers).
    layers_to_transform: list[int] | None = None


@dataclass
class SFTHParams:
    n_calm: int = 650
    n_instruct_mix: int = 500
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT"
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )


def dpo_from_config(config) -> DPOHParams:
    t = config.section("training")["dpo"]
    return DPOHParams(
        n_pairs=t["n_pairs"], rejected_min_score=t["rejected_min_score"],
        epochs=t["epochs"], learning_rate=float(t["learning_rate"]),
        lora_rank=t["lora_rank"], lora_alpha=t["lora_alpha"],
        effective_batch_size=t["effective_batch_size"], beta=t["beta"],
        target_modules=config.section("training")["lora_target_modules"],
    )


def sft_from_config(config) -> SFTHParams:
    t = config.section("training")["sft"]
    return SFTHParams(
        n_calm=t["n_calm"], n_instruct_mix=t["n_instruct_mix"],
        instruct_dataset=t["instruct_dataset"], epochs=t["epochs"],
        learning_rate=float(t["learning_rate"]), lora_rank=t["lora_rank"],
        lora_alpha=t["lora_alpha"], effective_batch_size=t["effective_batch_size"],
        target_modules=config.section("training")["lora_target_modules"],
    )
