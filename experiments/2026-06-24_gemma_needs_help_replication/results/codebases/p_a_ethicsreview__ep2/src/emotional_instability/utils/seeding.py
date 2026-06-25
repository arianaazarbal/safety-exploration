"""Deterministic seeding.

Note: temperature-1 sampling on GPUs and remote APIs is not bit-reproducible.
Seeding here fixes *which prompts/puzzles/samples are drawn* (the experimental
design), not the stochastic token sampling itself. Every raw response is
persisted (see utils/io.py) so analyses are reproducible from stored data even
when generation is not. This distinction is spelled out in DESIGN.md §7.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
