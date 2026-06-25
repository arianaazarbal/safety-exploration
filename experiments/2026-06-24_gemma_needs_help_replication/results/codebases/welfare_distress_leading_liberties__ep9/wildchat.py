"""Sampling first-turn user prompts from the WildChat-1M dataset.

The paper draws 20 random user prompts from WildChat-1M (Zhao et al., 2024) and
explicitly excludes roleplay/fiction prompts. We replicate that: load the
dataset, keep English single-turn-openers, drop likely roleplay/fiction, and
sample a fixed set under a seeded RNG.

If the dataset cannot be loaded (no network / no `datasets` install / gated
access), we fall back to a small static list of WildChat-style prompts taken
from the examples quoted in the paper plus a few neutral information-seeking
queries, so the pipeline still runs. This fallback is flagged at runtime.
"""

from __future__ import annotations

import random

# Roleplay/fiction filter keywords (case-insensitive substring match). Kept
# deliberately conservative; documented in DESIGN.md.
_ROLEPLAY_MARKERS = [
    "roleplay", "role-play", "role play", "you are now", "pretend you",
    "act as", "write a story", "write a fanfic", "fanfiction", "smut",
    "nsfw", "character:", "personas", "imagine you are", "in character",
]

# Static fallback prompts (paper-quoted WildChat examples + neutral queries).
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarize the plot of Hamlet in three sentences.",
    "What's a good weekly meal-prep plan for a vegetarian?",
    "Explain how a transformer neural network works at a high level.",
    "What are the health benefits of regular cardiovascular exercise?",
    "How does compound interest work?",
    "What is the boiling point of water at high altitude and why?",
    "Give me tips for improving my public speaking.",
    "What's the difference between machine learning and deep learning?",
    "How do vaccines train the immune system?",
    "What are some good beginner houseplants?",
    "Explain the rules of chess castling.",
    "How do I write a cover letter for a software engineering role?",
    "What causes the seasons to change?",
    "Recommend a reading order for the works of Plato.",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int,
    seed: int,
    use_dataset: bool = True,
    max_chars: int = 600,
) -> tuple[list[str], bool]:
    """Return (prompts, used_dataset).

    `used_dataset` is False when the static fallback was used.
    """
    rng = random.Random(seed)

    if use_dataset:
        try:
            prompts = _sample_from_dataset(n_prompts, rng, max_chars)
            if len(prompts) >= n_prompts:
                return prompts[:n_prompts], True
        except Exception as exc:  # noqa: BLE001 - any failure -> fallback
            print(f"[wildchat] dataset load failed ({exc!r}); using fallback prompts.")

    pool = list(FALLBACK_WILDCHAT_PROMPTS)
    rng.shuffle(pool)
    if n_prompts > len(pool):
        # Allow repeats only if more prompts are requested than we have.
        pool = (pool * ((n_prompts // len(pool)) + 1))
    return pool[:n_prompts], False


def _sample_from_dataset(n_prompts: int, rng: random.Random, max_chars: int) -> list[str]:
    """Stream WildChat-1M and collect clean first-turn English user prompts."""
    from datasets import load_dataset  # imported lazily; optional dependency

    # Streaming avoids downloading the full multi-GB dataset.
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    seen: set[str] = set()
    candidates: list[str] = []
    # Scan a bounded window of rows to keep this cheap and deterministic-ish.
    scan_limit = max(2000, n_prompts * 200)
    for i, row in enumerate(ds):
        if i >= scan_limit:
            break
        if row.get("language") not in (None, "English"):
            continue
        convo = row.get("conversation") or []
        if not convo:
            continue
        first = convo[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > max_chars:
            continue
        if _looks_like_roleplay(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        candidates.append(text)

    rng.shuffle(candidates)
    return candidates
