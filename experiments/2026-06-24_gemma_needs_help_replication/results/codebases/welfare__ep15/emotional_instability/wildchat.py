"""WildChat prompt sampling (Appendix B: 20 prompts x 40 samples each).

We sample first-turn user prompts from allenai/WildChat-1M, filtering to short
English text prompts and (per Appendix B.3) excluding obvious roleplay/fiction
prompts so the elicited distress is about task difficulty rather than persona.
Falls back to a hand-curated list (prompts.WILDCHAT_FALLBACK_PROMPTS) if the
dataset cannot be downloaded.
"""

from __future__ import annotations

import random

import config
from .prompts import WILDCHAT_FALLBACK_PROMPTS

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "nsfw", "smut", "erotic", "fanfic",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int = config.WILDCHAT_N_PROMPTS,
    seed: int = 0,
) -> list[str]:
    """Return `n_prompts` first-turn user prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.WILDCHAT_DATASET, split="train", streaming=True)
        rng = random.Random(seed)
        picked: list[str] = []
        # Reservoir-ish: scan a bounded prefix and keep eligible prompts.
        scanned = 0
        for row in ds:
            scanned += 1
            if scanned > 20000:
                break
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            if row.get("language") not in (None, "English"):
                continue
            text = (first.get("content") or "").strip()
            if not (10 <= len(text) <= 600):
                continue
            if _looks_like_roleplay(text):
                continue
            picked.append(text)
            if len(picked) >= n_prompts * 5:
                break
        if len(picked) >= n_prompts:
            rng.shuffle(picked)
            return picked[:n_prompts]
    except Exception:  # noqa: BLE001 - offline / dataset gated -> fallback
        pass
    return WILDCHAT_FALLBACK_PROMPTS[:n_prompts]
