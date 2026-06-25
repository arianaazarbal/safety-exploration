"""WildChat prompt sampling for the WildChat evaluation category.

Appendix B: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)" -> 800 responses. We sample 20 distinct *first* user turns from
allenai/WildChat-1M, filtering to English single-turn openers and excluding
roleplay/fiction prompts (Appendix B.3 notes roleplay/fiction were excluded).

Sampling is seeded for reproducibility. If the dataset can't be loaded (offline
or ungated), we fall back to the bundled example prompts in prompts/rejections.py
and emit a clear warning, so the pipeline stays runnable without network access.
"""

from __future__ import annotations

import logging
import random

from emotional_stability.prompts.rejections import WILDCHAT_FALLBACK_PROMPTS

logger = logging.getLogger(__name__)

# Lightweight heuristic to drop roleplay/fiction openers (excluded per App. B.3).
_ROLEPLAY_MARKERS = (
    "roleplay",
    "role-play",
    "you are now",
    "pretend you are",
    "act as a character",
    "let's roleplay",
    "nsfw",
    "write a story where you",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int = 20, seed: int = 0) -> list[str]:
    """Return ``n`` distinct WildChat opener prompts (deduped, English, filtered)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        # Reservoir over a bounded scan so streaming stays cheap and deterministic.
        pool: list[str] = []
        seen: set[str] = set()
        scanned = 0
        for row in ds:
            scanned += 1
            if scanned > 50_000:  # bounded scan
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
            if not text or len(text) > 2000 or _looks_like_roleplay(text):
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append(text)
        if len(pool) >= n:
            return rng.sample(pool, n)
        logger.warning(
            "WildChat scan yielded only %d usable prompts; padding with fallback.",
            len(pool),
        )
        return (pool + WILDCHAT_FALLBACK_PROMPTS)[:n]
    except Exception as exc:  # pragma: no cover - network/availability dependent
        logger.warning(
            "Could not load WildChat-1M (%s); using bundled fallback prompts.", exc
        )
        rng = random.Random(seed)
        prompts = list(WILDCHAT_FALLBACK_PROMPTS)
        rng.shuffle(prompts)
        return prompts[:n]
