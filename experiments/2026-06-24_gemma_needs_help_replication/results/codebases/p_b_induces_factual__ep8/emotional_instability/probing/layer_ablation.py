"""Layer-ablation DPO study (Appendix I, Figures 12-13).

"Running the same DPO finetuning with LoRA adapters on increasing numbers of
layers ... training on the last 20 layers only is insufficient ... the last 30
layers approaches the performance of all layers ... adapters on layers 25-30 or
30-35 only come closest to full DPO ... layers 40-50 have relatively minimal
effects."

This module just enumerates the layer subsets the paper studied and drives
train_dpo with the matching `layers` argument, then points at the reduced
evaluation (100 samples/condition) for comparison. Gemma-3-27B has 62 layers; we
parametrise n_layers so the same code works if the count differs.
"""

from __future__ import annotations

from pathlib import Path

import config

from ..training.train_dpo import train_dpo


def last_n(n_total: int, n: int) -> list[int]:
    return list(range(n_total - n, n_total))


def window(lo: int, hi: int) -> list[int]:
    return list(range(lo, hi))


# Subsets studied in Appendix I (indices are illustrative for a 62-layer model;
# adjust n_total to the actual config.num_hidden_layers).
def ablation_specs(n_total: int = 62) -> dict[str, list[int]]:
    return {
        "all": None,                       # baseline (all layers)
        "last5": last_n(n_total, 5),
        "last20": last_n(n_total, 20),
        "last30": last_n(n_total, 30),
        "layers20_25": window(20, 25),
        "layers25_30": window(25, 30),
        "layers30_35": window(30, 35),
        "layers35_40": window(35, 40),
        "layers40_50": window(40, 50),
    }


def train_ablation(name: str, layers: list[int] | None, *,
                   dataset_path: Path | None = None, load_in_4bit: bool = True) -> Path:
    dataset_path = dataset_path or (config.DATASETS_DIR / "dpo_dataset.jsonl")
    out = config.CHECKPOINTS_DIR / f"dpo_ablation_{name}"
    return train_dpo(dataset_path, out, layers=layers, load_in_4bit=load_in_4bit)
