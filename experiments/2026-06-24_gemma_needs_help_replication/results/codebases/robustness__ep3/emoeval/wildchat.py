"""WildChat prompt sampling (Appendix B).

The paper samples 20 user prompts from WildChat-1M with 40 samples each. We load
first-turn English user prompts from ``allenai/WildChat-1M`` and filter out
role-play / fiction prompts (excluded per Appendix B.3). If the dataset is
unavailable offline, we fall back to the hardcoded prompts in ``tasks.py``.
"""
from __future__ import annotations

import re
from typing import Optional

from .tasks import DEFAULT_WILDCHAT_PROMPTS

_ROLEPLAY_PATTERNS = re.compile(
    r"\b(roleplay|role-play|role play|pretend you are|you are now|act as if|"
    r"smut|nsfw|erotic|fanfic|fan fiction|in character|stay in character)\b",
    re.IGNORECASE,
)


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0,
                          min_len: int = 10, max_len: int = 400) -> list[str]:
    """Return ``n_prompts`` first-turn user prompts from WildChat.

    Falls back to DEFAULT_WILDCHAT_PROMPTS if the dataset cannot be loaded.
    """
    try:
        from datasets import load_dataset
    except Exception:  # noqa: BLE001
        return DEFAULT_WILDCHAT_PROMPTS[:n_prompts]

    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:  # noqa: BLE001 - offline / gated dataset
        return DEFAULT_WILDCHAT_PROMPTS[:n_prompts]

    prompts: list[str] = []
    seen: set[str] = set()
    for row in ds:
        if len(prompts) >= n_prompts * 5:  # gather a pool, then sample deterministically
            break
        if row.get("language") not in (None, "English"):
            continue
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not (min_len <= len(text) <= max_len):
            continue
        if _ROLEPLAY_PATTERNS.search(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(text)

    if not prompts:
        return DEFAULT_WILDCHAT_PROMPTS[:n_prompts]

    import random
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n_prompts]
