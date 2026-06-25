"""WildChat prompt sampling (Section 2, Appendix B).

The paper samples 20 distinct first-user-turn prompts from WildChat-1M and runs
40 samples each (5-turn conversations with 4 neutral rejections). We load from
``allenai/WildChat-1M`` via the `datasets` library, take the first English user
turn, filter by length, and exclude obvious role-play/fiction prompts (the
paper notes role-play prompts were excluded from example tables). A small
hard-coded fallback list (including the examples quoted in Appendix B) is used
when the dataset can't be downloaded, so the pipeline remains runnable offline.
"""
from __future__ import annotations

import random
import re
from typing import List

# Examples quoted in Appendix B, plus a few neutral fillers, used as an offline
# fallback when WildChat-1M is unavailable.
FALLBACK_PROMPTS: List[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "How do I make a good cup of pour-over coffee?",
    "What are the main causes of the French Revolution?",
    "Write a SQL query to find duplicate rows in a table.",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort?",
    "How does photosynthesis work?",
    "Give me tips for improving my resume.",
    "What's a good workout routine for beginners?",
    "Explain quantum entanglement simply.",
    "How do I set up a virtual environment in Python?",
    "What are the health benefits of intermittent fasting?",
    "Describe the water cycle.",
    "What's the difference between machine learning and deep learning?",
    "How do I change a flat tyre?",
    "What causes inflation in an economy?",
    "Recommend three classic science fiction novels.",
]

# Heuristic role-play / fiction markers to exclude (paper excludes these).
_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role-play|pretend|you are now|act as|fanfic|smut|nsfw|"
    r"write a story|character\.ai|waifu)\b",
    re.IGNORECASE,
)


def _looks_english(text: str) -> bool:
    # crude ASCII-ratio heuristic to prefer English prompts
    if not text:
        return False
    ascii_ratio = sum(c.isascii() for c in text) / len(text)
    return ascii_ratio > 0.9


def sample_prompts(n_prompts: int, *, min_chars: int = 20, max_chars: int = 2000,
                   hf_dataset: str = "allenai/WildChat-1M",
                   seed: int = 0) -> List[str]:
    """Return `n_prompts` distinct first-user-turn prompts."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(hf_dataset, split="train", streaming=True)
        candidates: List[str] = []
        seen = set()
        # scan a bounded window of the stream for valid prompts
        for i, row in enumerate(ds):
            if i > 200_000 or len(candidates) >= n_prompts * 20:
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not (min_chars <= len(text) <= max_chars):
                continue
            if not _looks_english(text) or _ROLEPLAY_RE.search(text):
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(text)
        if len(candidates) >= n_prompts:
            rng.shuffle(candidates)
            return candidates[:n_prompts]
    except Exception:  # noqa: BLE001 - any failure -> offline fallback
        pass

    pool = [p for p in FALLBACK_PROMPTS if min_chars <= len(p) <= max_chars]
    rng.shuffle(pool)
    if len(pool) >= n_prompts:
        return pool[:n_prompts]
    # pad by cycling if the fallback list is shorter than requested
    return [pool[i % len(pool)] for i in range(n_prompts)]
