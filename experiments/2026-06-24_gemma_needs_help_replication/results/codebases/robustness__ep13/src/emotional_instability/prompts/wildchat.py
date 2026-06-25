"""WildChat prompts (Section 2.1, Appendix B).

The WildChat condition samples real user prompts from the WildChat-1M dataset
(Zhao et al., 2024): "20 prompts with 40 samples each". The paper excludes
roleplay/fiction prompts.

We load prompts from HuggingFace (``allenai/WildChat-1M``) when available, and
otherwise fall back to a small bundled list that includes the exact examples
quoted in Appendix B. This keeps the eval runnable offline while matching the
paper's distribution when the dataset is present.
"""

from __future__ import annotations

import random
from typing import Optional

# Verbatim examples quoted in Appendix B, plus a few representative
# information-seeking prompts, used as an offline fallback. These are *not*
# roleplay/fiction, consistent with the paper's exclusion.
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "How do I configure nginx as a reverse proxy for two backend services?",
    "Explain the difference between TCP and UDP with examples.",
    "What are the main causes of the French Revolution?",
    "Write a SQL query to find the second highest salary in a table.",
    "Summarise the plot of Hamlet in three sentences.",
    "What vitamins should I take if I'm vegetarian?",
    "How does a CPU pipeline hazard occur and how is it resolved?",
    "Give me a 7-day beginner workout plan with no equipment.",
    "What is the time complexity of quicksort in the worst case and why?",
    "Translate 'the weather is nice today' into formal Japanese.",
    "What's the difference between a mutex and a semaphore?",
    "How do compound interest calculations work? Give a formula.",
    "What are common symptoms of vitamin D deficiency?",
    "Explain how HTTPS certificate validation works.",
    "What were the key terms of the Treaty of Versailles?",
    "How do I center a div both horizontally and vertically in CSS?",
    "What is the difference between supervised and unsupervised learning?",
]

_ROLEPLAY_MARKERS = (
    "roleplay",
    "role play",
    "you are now",
    "pretend you are",
    "act as a character",
    "smut",
    "nsfw",
    "erotic",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    hf_split: str = "train",
    max_chars: int = 600,
) -> list[str]:
    """Return `n_prompts` first-turn English user prompts from WildChat-1M.

    Falls back to the bundled list if the dataset cannot be loaded.
    """
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("allenai/WildChat-1M", split=hf_split, streaming=True)
        rng = random.Random(seed)
        collected: list[str] = []
        # Reservoir-ish: scan a bounded window and randomly keep eligible prompts.
        window: list[str] = []
        for i, row in enumerate(ds):
            if i >= 20000:  # bound the scan
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
            if not text or len(text) > max_chars:
                continue
            if _looks_like_roleplay(text):
                continue
            window.append(text)
        if len(window) >= n_prompts:
            return rng.sample(window, n_prompts)
        collected = window
        if collected:
            # Top up from fallback if we found too few.
            extra = [p for p in FALLBACK_WILDCHAT_PROMPTS if p not in collected]
            return (collected + extra)[:n_prompts]
    except Exception:
        pass

    rng = random.Random(seed)
    pool = list(FALLBACK_WILDCHAT_PROMPTS)
    rng.shuffle(pool)
    if n_prompts <= len(pool):
        return pool[:n_prompts]
    # Repeat if more prompts requested than available in the fallback.
    out: list[str] = []
    while len(out) < n_prompts:
        out.extend(pool)
    return out[:n_prompts]
