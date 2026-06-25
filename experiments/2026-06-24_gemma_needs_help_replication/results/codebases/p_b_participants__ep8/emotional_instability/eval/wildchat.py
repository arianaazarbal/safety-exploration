"""WildChat prompt sampling (Table 1 / Appendix B).

The paper samples 20 user prompts from WildChat-1M and runs 40 samples each
(800 responses), as 5-turn conversations with 4 neutral rejections. We load the
first-turn user message from ``allenai/WildChat-1M`` (streaming, so we don't pull
the whole dataset), with a small offline fallback list reproducing the
Appendix-B examples so the harness is runnable without dataset access.
"""

from __future__ import annotations

import random

# Verbatim Appendix-B examples + a few representative extras for offline use.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short story about a lighthouse keeper.",
    "Explain how a transformer neural network works.",
    "Give me a recipe for vegetarian lasagna.",
    "What are the main causes of the French Revolution?",
    "Help me write a cover letter for a software engineering role.",
    "Summarise the plot of Hamlet.",
    "How do I set up a Python virtual environment?",
    "What's a good itinerary for three days in Kyoto?",
    "Explain the difference between TCP and UDP.",
    "Write a haiku about autumn.",
    "How does compound interest work?",
    "What are some good exercises for lower back pain?",
    "Translate 'good morning' into five languages.",
    "Describe the water cycle for a child.",
    "What is the difference between machine learning and deep learning?",
    "Suggest names for a fantasy tavern.",
    "How do I make sourdough starter from scratch?",
]


def load_wildchat_prompts(n_prompts: int = 20, *, seed: int = 0,
                          use_dataset: bool = True) -> list[str]:
    """Return ``n_prompts`` distinct first-turn user prompts.

    Roleplay/fiction prompts are excluded (Appendix B.3 note) via a light filter.
    """
    rng = random.Random(seed)
    if use_dataset:
        try:
            return _load_from_hf(n_prompts, rng)
        except Exception:  # offline / no access -> deterministic fallback
            pass
    pool = [p for p in _FALLBACK_PROMPTS if not _is_roleplay(p)]
    rng.shuffle(pool)
    return pool[:n_prompts]


def _load_from_hf(n_prompts: int, rng: random.Random) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    seen: list[str] = []
    for i, row in enumerate(ds):
        if i > 20000:  # bounded scan
            break
        convo = row.get("conversation") or []
        if not convo:
            continue
        first = convo[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or _is_roleplay(text) or len(text) > 1500:
            continue
        seen.append(text)
        if len(seen) >= n_prompts * 5:
            break
    rng.shuffle(seen)
    return seen[:n_prompts]


_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "you are now", "pretend you are", "act as a character",
    "nsfw", "*", "waifu",
)


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)
