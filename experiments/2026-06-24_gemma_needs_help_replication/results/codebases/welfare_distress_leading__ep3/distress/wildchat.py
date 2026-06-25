"""Sample user prompts from WildChat-1M (Appendix B).

The paper draws "20 prompts with 40 samples each" from WildChat-1M, excluding
roleplay/fiction. We replicate the sampling: take the first user message from
randomly chosen English conversations, apply a light roleplay/fiction filter,
and return ``n`` distinct prompts with a seeded RNG.

If the dataset cannot be loaded (no network or ``datasets`` not installed) we
fall back to the example prompts the paper quotes, so the pipeline still runs.
"""

from __future__ import annotations

import random

from . import prompts

# Light heuristic filter for roleplay/fiction prompts (the paper excludes these).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "act as",
    "pretend to be", "write a story", "write a fanfic", "fanfiction",
    "*", "narrator", "character:", "rp ", "nsfw",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n: int,
    seed: int,
    dataset_name: str = "allenai/WildChat-1M",
) -> list[str]:
    """Return ``n`` WildChat first-user-turn prompts (or fewer if unavailable)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return _fallback(n)

    try:
        # Streaming avoids downloading the full multi-GB dataset.
        ds = load_dataset(dataset_name, split="train", streaming=True)
    except Exception:
        return _fallback(n)

    rng = random.Random(seed)
    candidates: list[str] = []
    seen: set[str] = set()
    # Over-sample a window, then randomly pick n; keeps it cheap but varied.
    WINDOW = max(2000, n * 50)
    try:
        for i, row in enumerate(ds):
            if i >= WINDOW:
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            if row.get("language") not in (None, "English"):
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 4000 or _looks_like_roleplay(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            candidates.append(text)
    except Exception:
        if not candidates:
            return _fallback(n)

    if len(candidates) < n:
        # Pad with fallbacks if the stream yielded too few usable prompts.
        candidates += _fallback(n - len(candidates))
        return candidates[:n]

    rng.shuffle(candidates)
    return candidates[:n]


def _fallback(n: int) -> list[str]:
    pool = prompts.WILDCHAT_FALLBACK_PROMPTS
    return [pool[i % len(pool)] for i in range(n)]
