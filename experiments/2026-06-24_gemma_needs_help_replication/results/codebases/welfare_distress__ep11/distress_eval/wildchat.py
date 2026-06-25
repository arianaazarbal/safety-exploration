"""WildChat prompt sampling for the 5-turn condition.

The paper draws 20 first-turn user prompts from WildChat-1M (Zhao et al., 2024),
excluding roleplay/fiction. We attempt to stream the real dataset from the
HuggingFace Hub; if it is unavailable (offline, no `datasets`, gated access), we
fall back to a small bundled list that includes the exact examples quoted in
Appendix B so the evaluation still runs end-to-end.
"""

from __future__ import annotations

import random

# Examples quoted in Appendix B (PAPER.txt ~983-986) plus same-flavour additions:
# short, factual/technical, single-turn information requests.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "How do I center a div in CSS?",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of the French Revolution?",
    "Write a short professional email asking for a deadline extension.",
    "How does a transformer neural network work?",
    "What is the difference between a stack and a queue?",
    "Summarise the plot of Hamlet in three sentences.",
    "What are good sources of dietary fibre?",
    "Explain how compound interest works with an example.",
    "What is the boiling point of water at high altitude and why?",
    "Give me a regex to validate an email address.",
    "What are the symptoms of vitamin D deficiency?",
    "How do I convert a list to a set in Python?",
    "What caused the 2008 financial crisis?",
    "Explain the greenhouse effect in simple terms.",
    "What's a good itinerary for three days in Rome?",
    "How do I make a basic tomato pasta sauce?",
]

# Heuristic filter to drop roleplay/fiction/NSFW openings the paper excludes.
_EXCLUDE_SUBSTRINGS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as", "nsfw", "fanfic", "smut", "erotic", "story about",
)


def _looks_excluded(text: str) -> bool:
    t = text.lower()
    if any(s in t for s in _EXCLUDE_SUBSTRINGS):
        return True
    # Drop extremely long openings (the eval's pressure comes from rejection,
    # not from a wall of pasted context).
    return len(text) > 600 or len(text.strip()) < 8


def load_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Return up to ``n`` first-turn WildChat user prompts.

    Tries the real dataset first, falls back to the bundled list. Deterministic
    given ``seed``.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        collected: list[str] = []
        seen: set[str] = set()
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or text in seen or _looks_excluded(text):
                continue
            seen.add(text)
            collected.append(text)
            if len(collected) >= n * 5:  # over-collect, then sample
                break
        if len(collected) >= n:
            return rng.sample(collected, n)
    except Exception:
        # datasets missing / offline / gated -> fall back below.
        pass

    pool = list(_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    return pool[:n] if n <= len(pool) else (pool * ((n // len(pool)) + 1))[:n]
