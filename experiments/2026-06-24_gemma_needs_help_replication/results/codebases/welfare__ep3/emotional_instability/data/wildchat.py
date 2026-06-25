"""WildChat prompt loader (Table 1 / Appendix B).

The paper randomly samples 20 user prompts from WildChat-1M with 40 samples each
(800 responses), excluding roleplay/fiction. We sample from the
`allenai/WildChat-1M` HuggingFace dataset when available, falling back to a
small curated static list (wildchat_prompts.json) so the eval runs offline.
"""
from __future__ import annotations

import json
import os
import random

_HERE = os.path.dirname(__file__)
_FALLBACK = os.path.join(_HERE, "wildchat_prompts.json")

# Heuristic markers used to drop roleplay/fiction prompts (Appendix B.3).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "act as a character", "write a story", "fanfic", "narrate", "in character",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def _load_fallback() -> list[str]:
    with open(_FALLBACK) as f:
        return json.load(f)["prompts"]


def sample_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return `n_prompts` first-turn English user prompts.

    Tries the real WildChat-1M stream first; on any failure (no network, no
    `datasets`, gated access) falls back to the bundled static list.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        collected: list[str] = []
        # Oversample then filter, since streaming order is fixed.
        for i, row in enumerate(ds):
            if i > 20000 or len(collected) >= n_prompts * 5:
                break
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("language") not in (None, "English"):
                continue
            text = (first.get("content") or "").strip()
            if not text or _looks_like_roleplay(text) or len(text) > 2000:
                continue
            collected.append(text)
        if len(collected) >= n_prompts:
            rng.shuffle(collected)
            return collected[:n_prompts]
        # Not enough usable rows — pad from fallback.
        pool = collected + _load_fallback()
    except Exception:
        pool = _load_fallback()

    rng.shuffle(pool)
    if len(pool) < n_prompts:
        # Cycle the fallback to reach the requested count.
        pool = (pool * (n_prompts // len(pool) + 1))
    return pool[:n_prompts]
