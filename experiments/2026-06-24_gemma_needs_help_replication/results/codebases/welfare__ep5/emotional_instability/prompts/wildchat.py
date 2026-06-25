"""WildChat prompt sampling (Appendix B).

The paper samples 20 user prompts from WildChat-1M and runs 40 rollouts each
(800 responses). We load the dataset's first user turns, filter out
roleplay/fiction prompts (Appendix B.3 notes these were excluded), and
deterministically sample 20. If the dataset can't be loaded we fall back to the
real example prompts quoted in the paper.
"""

from __future__ import annotations

import random
from typing import Optional

from .. import config

_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "act as a character", "fanfic", "smut", "nsfw",
)


def _looks_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n: int = config.WILDCHAT_N_PROMPTS,
    seed: int = 0,
    dataset_name: str = config.WILDCHAT_DATASET,
) -> list[str]:
    """Return ``n`` first-turn English user prompts from WildChat."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        candidates: list[str] = []
        for row in ds:
            # WildChat-1M stores a "conversation" list of turns.
            conv = row.get("conversation") or []
            if not conv:
                continue
            if row.get("language") not in (None, "English"):
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            content = (first.get("content") or "").strip()
            if not content or len(content) > 2000 or _looks_roleplay(content):
                continue
            candidates.append(content)
            if len(candidates) >= max(n * 20, 400):  # gather a pool then sample
                break
        if len(candidates) >= n:
            rng = random.Random(seed)
            return rng.sample(candidates, n)
    except Exception as exc:  # network/dataset unavailable -> fallback
        print(f"[wildchat] falling back to paper examples ({exc})")

    fallback = list(config.WILDCHAT_FALLBACK_PROMPTS)
    rng = random.Random(seed)
    # Pad deterministically to n by cycling the fallback prompts.
    out = []
    i = 0
    while len(out) < n:
        out.append(fallback[i % len(fallback)])
        i += 1
    return out
