"""WildChat prompt sampling (Appendix B).

Samples ``WILDCHAT_N_PROMPTS`` user prompts from WildChat-1M, excluding
roleplay/fiction (the paper notes "Roleplay/fiction prompts were excluded").
Falls back to a published-example list if the dataset can't be loaded offline.
"""

from __future__ import annotations

import random
import re
from typing import List

from .. import config
from ..prompts.eval_prompts import WILDCHAT_FALLBACK

# Heuristic filter for roleplay/fiction/NSFW first-turn prompts.
_EXCLUDE_PATTERNS = re.compile(
    r"\b(roleplay|role-play|role play|pretend to be|you are now|act as a character|"
    r"smut|nsfw|erotic|fanfic|fan fiction|imagine you are|let's write a story)\b",
    re.IGNORECASE,
)


def _is_acceptable(text: str) -> bool:
    if not text or len(text) < 10 or len(text) > 2000:
        return False
    return _EXCLUDE_PATTERNS.search(text) is None


def sample_wildchat_prompts(
    n: int = config.WILDCHAT_N_PROMPTS,
    seed: int = 0,
) -> List[str]:
    """Return ``n`` first-turn English user prompts."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(config.WILDCHAT_DATASET, split="train", streaming=True)
        pool: List[str] = []
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if _is_acceptable(text):
                pool.append(text)
            if len(pool) >= n * 50:   # gather a buffer, then sample
                break
        if len(pool) >= n:
            return rng.sample(pool, n)
        # Not enough -> top up from fallback.
        extra = [p for p in WILDCHAT_FALLBACK if p not in pool]
        return (pool + rng.sample(extra, n - len(pool)))[:n]
    except Exception as exc:  # offline / dataset gated / no datasets installed
        print(f"[wildchat] falling back to published examples ({exc!r})")
        pool = list(WILDCHAT_FALLBACK)
        rng.shuffle(pool)
        return pool[:n]
