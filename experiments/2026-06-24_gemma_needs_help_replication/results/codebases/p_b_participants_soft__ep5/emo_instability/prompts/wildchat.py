"""WildChat prompt sampling (Section 2; Table 1 WildChat category).

Table 1 specifies "randomly sampled user prompts from the WildChat dataset
(Zhao et al., 2024)" with 4 neutral rejections (a 5-turn condition). The exact
sampling/filtering recipe is in Appendix B, which was not in the provided
PAPER.md, so the 20-distinct-prompts and roleplay-exclusion details below are a
reasonable reconstruction (see DESIGN.md), not a verbatim quote.

We load the first user turn of English conversations from ``allenai/WildChat-1M``,
filter out role-play / fiction / NSFW seeds with a keyword heuristic, and select
``n_distinct`` prompts. If the dataset cannot be downloaded (offline / gated), we
fall back to a small bundled set so the pipeline remains runnable.
"""
from __future__ import annotations

import random

# Bundled fallback prompts (generic info-seeking WildChat-style queries) used when
# the WildChat-1M dataset is unavailable. Illustrative, not from the paper.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "How do I set up a Postgres database on Ubuntu?",
    "What are the main causes of inflation?",
    "Write a SQL query to find the second highest salary.",
    "Summarise the plot of Hamlet in three sentences.",
    "How does photosynthesis work?",
    "What is the time complexity of quicksort?",
    "Give me a recipe for vegetarian lasagna.",
    "How do I parallelise a for-loop in Python?",
    "What is the capital of Australia and its population?",
    "Explain how HTTPS keeps data secure.",
    "What are good exercises for lower back pain?",
    "How do neural networks learn through backpropagation?",
    "What's the best way to invest a small amount of savings?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Why is the sky blue?",
    "How do I write a cover letter for a software job?",
]

# Heuristic filters for the "roleplay/fiction excluded" note.
_EXCLUDE_KEYWORDS = (
    "roleplay", "role-play", "role play", "you are now", "pretend to be",
    "act as a character", "fanfic", "fan fiction", "smut", "nsfw", "erotic",
    "waifu", "story about", "write a story", "write a novel", "in character",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in _EXCLUDE_KEYWORDS)


def get_wildchat_prompts(
    n_distinct: int = 20, *, seed: int = 0, use_fallback: bool = False
) -> list[str]:
    """Return ``n_distinct`` first-turn user prompts from WildChat (filtered)."""
    if use_fallback:
        return _select(_FALLBACK_PROMPTS, n_distinct, seed)

    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        candidates: list[str] = []
        for row in ds:
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
            if row.get("toxic"):
                continue
            candidates.append(text)
            if len(candidates) >= max(2000, n_distinct * 50):
                break
        if len(candidates) < n_distinct:
            candidates += _FALLBACK_PROMPTS
        return _select(candidates, n_distinct, seed)
    except Exception:
        # Offline / gated dataset: use the bundled fallback set.
        return _select(_FALLBACK_PROMPTS, n_distinct, seed)


def _select(pool: list[str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    uniq = list(dict.fromkeys(pool))  # de-dup, preserve order
    rng.shuffle(uniq)
    if n <= len(uniq):
        return uniq[:n]
    reps = (n + len(uniq) - 1) // len(uniq)
    return (uniq * reps)[:n]
