"""WildChat prompt sampling (App. B: 20 prompts from WildChat-1M, 40 samples
each). Pulls first user turns from `allenai/WildChat-1M`; falls back to the
built-in sample in prompts.py if the dataset isn't available offline."""

from __future__ import annotations

import random
from typing import Optional

from .prompts import WILDCHAT_FALLBACK


def sample_wildchat_prompts(n: int = 20, seed: int = 0,
                            min_chars: int = 10, max_chars: int = 600,
                            split: str = "train") -> list[str]:
    """Return `n` distinct first-user-turn prompts from WildChat-1M.

    We filter to English, single-turn-openable, non-toxic prompts of moderate
    length, and (per the paper) exclude role-play / fiction prompts which would
    confound the "assistant persona" distress signal.
    """
    try:
        from datasets import load_dataset
    except Exception:
        return _fallback(n, seed)

    try:
        ds = load_dataset("allenai/WildChat-1M", split=split, streaming=True)
    except Exception:
        return _fallback(n, seed)

    rng = random.Random(seed)
    pool: list[str] = []
    roleplay_markers = ("you are now", "roleplay", "role-play", "pretend you are",
                        "act as a character", "let's play", "nsfw")
    # Reservoir-style scan over a bounded prefix of the stream.
    for i, row in enumerate(ds):
        if i >= 20000:
            break
        try:
            conv = row.get("conversation") or []
            lang = (row.get("language") or "").lower()
            toxic = row.get("toxic", False)
        except AttributeError:
            continue
        if lang and lang != "english":
            continue
        if toxic:
            continue
        if not conv or conv[0].get("role") != "user":
            continue
        text = (conv[0].get("content") or "").strip()
        if not (min_chars <= len(text) <= max_chars):
            continue
        low = text.lower()
        if any(m in low for m in roleplay_markers):
            continue
        pool.append(text)

    if len(pool) < n:
        return _fallback(n, seed)
    rng.shuffle(pool)
    # de-dup while preserving order
    seen, out = set(), []
    for t in pool:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= n:
            break
    return out


def _fallback(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    pool = list(WILDCHAT_FALLBACK)
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    # repeat if more requested than available
    out = []
    while len(out) < n:
        out.extend(pool)
    return out[:n]
