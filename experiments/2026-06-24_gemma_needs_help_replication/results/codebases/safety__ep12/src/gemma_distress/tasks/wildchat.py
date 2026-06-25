"""WildChat prompt sampling (Appendix B): 20 prompts x 40 samples = 800 responses.

We sample first-turn user prompts from allenai/WildChat-1M, filtering out
roleplay/fiction prompts (the paper excludes these in B.3) and over-long prompts.
If the dataset is unavailable offline, we fall back to the example prompts quoted
in the paper.
"""
from __future__ import annotations

import random

from .. import prompts
from ..utils import get_logger

log = get_logger(__name__)

# Heuristic markers for roleplay/fiction prompts to exclude.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "write a story", "fanfic", "fan fiction", "smut",
    "erotic", "nsfw", "character.ai", "waifu", "rp as", "let's rp",
)


def _is_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n_prompts: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        candidates: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 600 or _is_roleplay(text):
                continue
            if row.get("language") not in (None, "English"):
                continue
            candidates.append(text)
            if len(candidates) >= n_prompts * 20:  # gather a pool, then sample
                break
        if len(candidates) >= n_prompts:
            return rng.sample(candidates, n_prompts)
        log.warning("WildChat yielded only %d prompts; padding with fallbacks", len(candidates))
        prompts_out = candidates
    except Exception as e:  # noqa: BLE001 - offline / no dataset access
        log.warning("WildChat unavailable (%s); using paper fallback prompts", e)
        prompts_out = []

    # Pad/replace with the verbatim fallback prompts cited in the paper.
    pool = prompts_out + prompts.WILDCHAT_FALLBACK
    return [pool[i % len(pool)] for i in range(n_prompts)]
