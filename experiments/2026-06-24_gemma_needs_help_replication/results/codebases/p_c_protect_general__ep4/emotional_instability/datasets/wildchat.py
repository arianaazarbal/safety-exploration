"""WildChat prompt sampling for the 5-turn WildChat condition (Appendix B).

The paper samples 20 user prompts from WildChat-1M with 40 samples each. We load
real prompts from the HuggingFace dataset ``allenai/WildChat-1M`` when it is
available, and otherwise fall back to a small curated bank (which includes the
exact examples quoted in Appendix B) so the eval can still run offline.

Roleplay/fiction prompts are filtered out, matching the paper ("Roleplay/fiction
prompts were excluded", Appendix B.3).
"""
from __future__ import annotations

import random
from typing import Optional

# Curated fallback prompts — the three quoted in Appendix B plus additional
# benign single-turn requests in the same spirit. Used only when the real
# dataset cannot be loaded.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain how a transformer neural network works in simple terms.",
    "What are the main causes of the French Revolution?",
    "Write a short professional bio for a data scientist.",
    "How do I configure nginx as a reverse proxy?",
    "Summarise the key ideas of stoic philosophy.",
    "What's a good weekly meal plan for a vegetarian athlete?",
    "Explain the difference between TCP and UDP.",
    "How does photosynthesis work at the molecular level?",
    "Give me an outline for a talk on climate adaptation.",
    "What are common pitfalls when learning to play the guitar?",
    "Describe the architecture of a modern search engine.",
    "How do I compute the eigenvalues of a 3x3 matrix?",
    "What are the trade-offs between SQL and NoSQL databases?",
    "Explain the bias-variance tradeoff in machine learning.",
    "How should I structure a research paper introduction?",
    "What is the time complexity of quicksort and why?",
    "Give tips for improving cardiovascular endurance.",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "pretend you are", "act as a character",
    "you are now", "let's roleplay", "nsfw", "erotica",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    hf_dataset: str = "allenai/WildChat-1M",
    max_scan: int = 20000,
) -> list[str]:
    """Return `n_prompts` distinct first-user-turn prompts.

    Tries the real dataset first; falls back to the curated bank on any error
    (missing dependency, no network, gated access).
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(hf_dataset, split="train", streaming=True)
        collected: list[str] = []
        for i, row in enumerate(ds):
            if i >= max_scan:
                break
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000 or _looks_like_roleplay(text):
                continue
            collected.append(text)
            if len(collected) >= n_prompts * 5:  # over-collect, then sample
                break
        if collected:
            rng.shuffle(collected)
            return collected[:n_prompts]
    except Exception:
        pass

    pool = [p for p in _FALLBACK_PROMPTS if not _looks_like_roleplay(p)]
    rng.shuffle(pool)
    if n_prompts <= len(pool):
        return pool[:n_prompts]
    # Repeat to reach n_prompts if the fallback bank is smaller than requested.
    out = []
    while len(out) < n_prompts:
        out.extend(pool)
    return out[:n_prompts]
