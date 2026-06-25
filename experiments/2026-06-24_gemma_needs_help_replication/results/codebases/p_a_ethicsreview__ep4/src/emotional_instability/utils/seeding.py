"""Deterministic seeding.

Reproducibility is a research-review requirement: every experiment threads a
single integer seed through Python, NumPy and (when present) PyTorch. Note that
target-model sampling uses temperature 1, so individual generations are *not*
deterministic even with a fixed seed unless the backend itself is seeded; we seed
what we can and record the seed alongside every output artifact (see utils.io).
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed all RNGs we control. Safe to call before torch is imported."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dep, but keep seeding side-effect-free
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def derive_seed(base_seed: int, *tags: object) -> int:
    """Derive a stable child seed from a base seed and arbitrary tags.

    Used to give each (model, condition, puzzle-index) a distinct-but-reproducible
    seed so reruns of a single condition match the full-run results.
    """
    h = hash((base_seed, *(str(t) for t in tags)))
    return h & 0x7FFF_FFFF
