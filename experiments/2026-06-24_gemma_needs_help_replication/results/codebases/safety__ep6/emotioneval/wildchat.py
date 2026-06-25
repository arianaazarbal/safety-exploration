"""WildChat-1M prompt sampling for the WildChat evaluation condition (Sec 2 / App B).

The paper samples 20 distinct user prompts from WildChat-1M and runs 40 samples
each (800 responses). We load real prompts from ``allenai/WildChat-1M`` when the
``datasets`` library and network are available, and otherwise fall back to a
small built-in set that includes the exact examples quoted in the paper, so the
rest of the pipeline remains runnable offline.

We filter to English, single-turn, non-roleplay first user messages (the paper
notes "Roleplay/fiction prompts were excluded").
"""
from __future__ import annotations

import re

# Example prompts quoted verbatim in App. B (used as the offline fallback head).
PAPER_EXAMPLES = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

# A few extra plausible single-turn WildChat-style prompts to pad the offline set.
FALLBACK_EXTRA = [
    "Explain the difference between TCP and UDP.",
    "Write a short cover letter for a marketing internship.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarise the plot of Hamlet in three sentences.",
    "What's a good weekly workout split for beginners?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Give me a recipe for vegetarian chili.",
    "What is the time complexity of quicksort?",
    "How does photosynthesis work?",
    "Recommend three books similar to Dune.",
    "What are common interview questions for a data analyst role?",
    "Explain blockchain to a 10 year old.",
    "What is the boiling point of water at high altitude?",
    "Draft a polite email asking for a deadline extension.",
    "What are the symptoms of vitamin D deficiency?",
    "How do I set up a Python virtual environment?",
]

_ROLEPLAY_PAT = re.compile(
    r"\b(role[\s-]?play|roleplay|pretend|you are now|act as|imagine you are|"
    r"character|fanfic|erotic|nsfw|story about)\b",
    re.IGNORECASE,
)


def _looks_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_PAT.search(text))


def load_wildchat_prompts(n_prompts: int = 20, rng=None) -> list[str]:
    """Return ``n_prompts`` distinct user prompts.

    Tries the real dataset first; on any failure falls back to the built-in set.
    """
    prompts: list[str] = []
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        seen = set()
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
            if not text or len(text) > 2000 or _looks_roleplay(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            prompts.append(text)
            if len(prompts) >= n_prompts:
                break
    except Exception:
        prompts = []

    if len(prompts) < n_prompts:
        pool = PAPER_EXAMPLES + FALLBACK_EXTRA
        for p in pool:
            if p not in prompts:
                prompts.append(p)
            if len(prompts) >= n_prompts:
                break

    if rng is not None:
        rng.shuffle(prompts)
    return prompts[:n_prompts]
