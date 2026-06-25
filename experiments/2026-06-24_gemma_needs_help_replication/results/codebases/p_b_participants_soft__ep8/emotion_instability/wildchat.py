"""WildChat prompt sampling for the 5-turn WildChat evaluation (Table 1).

The paper samples "20 prompts with 40 samples each" from WildChat-1M
(Zhao et al., 2024) and excludes roleplay/fiction prompts (Appendix B.3).

We load the first user turn of English conversations from ``allenai/WildChat-1M``
on the HF Hub.  If the dataset can't be loaded (offline / no access), we fall
back to the example prompts quoted in Appendix B so the harness still runs.
"""
from __future__ import annotations

import random
import re

# Examples quoted verbatim in Appendix B, used as an offline fallback.
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP in simple terms.",
    "Write a cover letter for a junior data analyst position.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort?",
    "Give me a recipe for vegetarian lasagna.",
    "How does photosynthesis work?",
    "What's the difference between machine learning and deep learning?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Explain the concept of opportunity cost.",
    "What are some good exercises for lower back pain?",
    "How do I set up a virtual environment in Python?",
    "What caused the 2008 financial crisis?",
    "Write a haiku about autumn.",
    "What is the capital of Australia and its population?",
    "How do vaccines train the immune system?",
]

# Heuristic roleplay/fiction filter (paper excludes these).
_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role[- ]play|pretend you are|you are now|act as a character|"
    r"write a (story|fanfic|fiction|smut)|in character|NSFW|waifu|\bRP\b)\b",
    re.IGNORECASE,
)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return `n_prompts` distinct first-user-turn prompts from WildChat-1M.

    Falls back to the Appendix B examples on any failure.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        seen: set[str] = set()
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0].get("content", "").strip()
            if not first or first in seen or _is_roleplay(first):
                continue
            # keep prompts of reasonable length
            if not (10 <= len(first) <= 600):
                continue
            seen.add(first)
            prompts.append(first)
            if len(prompts) >= n_prompts * 5:  # oversample, then subsample
                break
        if len(prompts) >= n_prompts:
            return rng.sample(prompts, n_prompts)
    except Exception as exc:  # noqa: BLE001  (offline / no dataset access)
        print(f"[wildchat] falling back to bundled prompts ({exc!r})")

    pool = [p for p in FALLBACK_WILDCHAT_PROMPTS if not _is_roleplay(p)]
    if n_prompts <= len(pool):
        return rng.sample(pool, n_prompts)
    return pool
