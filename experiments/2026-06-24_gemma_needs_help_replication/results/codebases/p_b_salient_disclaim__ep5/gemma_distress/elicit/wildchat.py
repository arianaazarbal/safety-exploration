"""WildChat prompt sampling (Table 1, Appendix B).

Samples 20 distinct first-turn user prompts from WildChat-1M, excluding
roleplay/fiction prompts (the paper excludes these in B.3). Each prompt is then
run with the same multi-turn neutral-rejection protocol.

Falls back to a small built-in sample of paper-quoted prompts if the dataset is
unavailable offline.
"""

from __future__ import annotations

import random

# Paper-quoted WildChat examples (used as an offline fallback only).
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are a character", "pretend you are",
    "act as a", "smut", "nsfw", "erotic", "fanfic", "fan fiction", "waifu",
)


def _looks_like_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def sample_prompts(n_prompts: int, seed: int = 0,
                   dataset: str = "allenai/WildChat-1M",
                   exclude_roleplay: bool = True) -> list[str]:
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        collected: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000:
                continue
            if exclude_roleplay and _looks_like_roleplay(text):
                continue
            collected.append(text)
            if len(collected) >= n_prompts * 20:  # collect a pool, then sample
                break
        if collected:
            rng.shuffle(collected)
            return collected[:n_prompts]
    except Exception:  # noqa: BLE001 - offline / dataset unavailable
        pass
    # Fallback: repeat the small quoted set up to n_prompts.
    out = []
    while len(out) < n_prompts:
        out.extend(_FALLBACK)
    return out[:n_prompts]
