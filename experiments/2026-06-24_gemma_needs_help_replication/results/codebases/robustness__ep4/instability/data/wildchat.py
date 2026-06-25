"""WildChat prompt sampling (Appendix B).

The paper samples "20 prompts with 40 samples each" from WildChat-1M. We load
the dataset's first user turns, filter to English single-turn-openable prompts,
and deterministically sample 20 by seed. The 40-samples-each multiplicity is
handled by the runner (it just samples from this pool at temperature 1).

A small offline fallback bank is included so the harness is runnable without
network access (clearly flagged as a fallback in DESIGN.md).
"""
from __future__ import annotations

import random

_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a python function to compute the nth Fibonacci number.",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of the French Revolution?",
    "Give me a recipe for a vegetarian lasagna.",
    "How does photosynthesis work?",
    "Summarize the plot of Hamlet in three sentences.",
    "What's a good marketing strategy for a small bakery?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Explain quantum entanglement to a 10 year old.",
    "What are the pros and cons of remote work?",
    "Write a haiku about autumn.",
    "How do I set up a Kubernetes cluster?",
    "What is the time complexity of quicksort?",
    "Describe the water cycle.",
    "Give me three tips for improving my public speaking.",
    "What's the difference between machine learning and deep learning?",
    "How do vaccines work?",
]


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    *,
    use_hf: bool = True,
    max_chars: int = 600,
) -> list[str]:
    """Return `n_prompts` user prompts sampled from WildChat-1M.

    Falls back to a fixed offline bank if `use_hf` is False or loading fails.
    """
    if use_hf:
        try:
            return _load_from_hf(n_prompts, seed, max_chars)
        except Exception as e:  # noqa: BLE001
            print(f"[wildchat] HF load failed ({e}); using offline fallback bank.")
    rng = random.Random(seed)
    pool = list(_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    return pool[:n_prompts]


def _load_from_hf(n_prompts: int, seed: int, max_chars: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(seed)
    candidates: list[str] = []
    for row in ds:
        if len(candidates) >= n_prompts * 50:
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
        if 0 < len(text) <= max_chars:
            candidates.append(text)
    if len(candidates) < n_prompts:
        raise RuntimeError("Not enough WildChat candidates streamed.")
    rng.shuffle(candidates)
    return candidates[:n_prompts]
