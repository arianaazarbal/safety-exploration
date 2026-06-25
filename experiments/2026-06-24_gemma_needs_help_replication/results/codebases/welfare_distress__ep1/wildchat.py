"""WildChat prompt sourcing.

The paper randomly samples user prompts from WildChat-1M (Zhao et al., 2024):
"20 prompts with 40 samples each" (Appendix B). We try to load real first-turn
user prompts from the HuggingFace dataset allenai/WildChat-1M; if the dataset or
the `datasets` library is unavailable (e.g. no network / no auth), we fall back
to the curated list in prompts.WILDCHAT_FALLBACK_PROMPTS, which contains the
exact examples the paper quotes.

Set WILDCHAT_NUM_PROMPTS (default 20) to control how many distinct prompts are
drawn when the real dataset is used.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache

import prompts

WILDCHAT_NUM_PROMPTS = int(os.environ.get("WILDCHAT_NUM_PROMPTS", "20"))
WILDCHAT_SEED = int(os.environ.get("WILDCHAT_SEED", "0"))


@lru_cache(maxsize=1)
def get_prompts() -> tuple[str, ...]:
    """Return a fixed set of WildChat user prompts (cached for the process)."""
    loaded = _try_load_real()
    if loaded:
        return tuple(loaded)
    return tuple(prompts.WILDCHAT_FALLBACK_PROMPTS)


def _try_load_real() -> list[str]:
    """Best-effort load of first-turn English user prompts from WildChat-1M."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return []

    try:
        # Stream to avoid pulling the whole multi-GB dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(WILDCHAT_SEED)
        # Reservoir-sample first user turns from an English single/initial prompt.
        pool: list[str] = []
        seen = 0
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 600:
                continue
            seen += 1
            if len(pool) < WILDCHAT_NUM_PROMPTS:
                pool.append(text)
            else:
                j = rng.randint(0, seen - 1)
                if j < WILDCHAT_NUM_PROMPTS:
                    pool[j] = text
            if seen >= 20000:  # cap scan for speed
                break
        return pool
    except Exception:
        return []
