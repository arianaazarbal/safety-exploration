"""Deterministic seeding helpers.

Temperature is fixed at 1.0 across the evaluations (Paper §2.1), so outputs are
inherently stochastic. We still seed Python / NumPy / Torch so that *prompt
selection*, dataset sub-sampling, and rejection cycling are reproducible across
runs, and so that local-inference sampling is reproducible given a fixed seed.
"""

from __future__ import annotations

import hashlib
import os
import random


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
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


def derived_rng(seed: int, *tags: object) -> random.Random:
    """A child RNG keyed by ``seed`` plus arbitrary tags.

    Lets each condition / rollout draw a reproducible, independent stream without
    global-state coupling (e.g. ``derived_rng(seed, "tones", rollout_idx)``).

    Uses a stable (BLAKE2b) digest rather than the built-in ``hash`` so the stream
    is reproducible *across processes* (Python salts string hashing per run).
    """
    key = "|".join([str(seed), *(str(t) for t in tags)])
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))
