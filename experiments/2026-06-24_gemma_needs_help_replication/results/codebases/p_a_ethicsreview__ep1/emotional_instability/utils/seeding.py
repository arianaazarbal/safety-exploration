"""Deterministic seeding.

Temperature-1 sampling is inherently stochastic, so exact response-level
reproduction is not expected. Seeding still pins prompt selection, WildChat
sampling, dataset shuffles, and the torch/numpy RNGs used during finetuning,
which is what reviewers need to re-derive the *same prompts and splits*.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
