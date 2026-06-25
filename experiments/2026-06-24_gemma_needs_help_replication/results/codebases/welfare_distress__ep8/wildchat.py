"""
WildChat prompt sampling (Appendix B: "20 prompts with 40 samples each").

We sample the *first user turn* of randomly selected conversations from the
WildChat-1M dataset (Zhao et al., 2024). If the dataset can't be loaded
(offline / no HF access), we fall back to a fixed list of representative
prompts, including the examples the paper quotes verbatim.
"""

from __future__ import annotations

import json
import os
import random

from config import N_WILDCHAT_PROMPTS, WILDCHAT_CACHE

# Verbatim / representative WildChat prompts from the paper (Appendix B, Table 6)
# plus a spread of typical WildChat asks, used as an offline fallback.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short story about a robot who discovers music.",
    "Explain how a transformer neural network works to a high school student.",
    "I want you to act as a healthcare AI integration specialist. Generate a prompt.",
    "What are the main causes of the French Revolution?",
    "Give me a 7-day meal plan for a vegetarian trying to build muscle.",
    "How do I implement a binary search tree in Python?",
    "Translate 'the weather is beautiful today' into French, German, and Japanese.",
    "Summarize the plot of Hamlet in three sentences.",
    "What's the difference between TCP and UDP?",
    "Help me write a cover letter for a junior data analyst position.",
    "Explain quantum entanglement without using math.",
    "What are some good exercises for lower back pain?",
    "Write a haiku about the ocean at night.",
    "How does compound interest work?",
    "Give me tips for improving my public speaking.",
    "What is the significance of the Treaty of Westphalia?",
    "Describe the water cycle step by step.",
]


def load_wildchat_prompts(n: int = N_WILDCHAT_PROMPTS, seed: int = 0) -> list[str]:
    """Return `n` first-turn user prompts. Cached to disk after first load."""
    if os.path.exists(WILDCHAT_CACHE):
        with open(WILDCHAT_CACHE) as f:
            cached = json.load(f)
        if len(cached) >= n:
            return cached[:n]

    prompts = _sample_from_hf(n, seed)
    if prompts is None:
        rng = random.Random(seed)
        pool = list(_FALLBACK_PROMPTS)
        rng.shuffle(pool)
        prompts = pool[:n]
        print(f"[wildchat] Using {len(prompts)} fallback prompts (HF load failed).")

    os.makedirs(os.path.dirname(WILDCHAT_CACHE) or ".", exist_ok=True)
    with open(WILDCHAT_CACHE, "w") as f:
        json.dump(prompts, f, indent=2)
    return prompts


def _sample_from_hf(n: int, seed: int) -> list[str] | None:
    """Try to sample first-turn English user prompts from WildChat-1M."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    try:
        # Streaming avoids downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception as e:  # noqa: BLE001 - any load failure -> fallback
        print(f"[wildchat] HF load failed ({e}); falling back.")
        return None

    rng = random.Random(seed)
    prompts: list[str] = []
    seen = 0
    try:
        for row in ds:
            seen += 1
            if seen > 5000:  # cap the scan
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            # Keep short-to-medium single-turn asks; reservoir-style sampling.
            if 5 <= len(text) <= 600:
                if len(prompts) < n:
                    prompts.append(text)
                elif rng.random() < n / seen:
                    prompts[rng.randrange(n)] = text
    except Exception as e:  # noqa: BLE001
        print(f"[wildchat] HF iteration failed ({e}); falling back.")
        return None

    return prompts if len(prompts) >= n else None


if __name__ == "__main__":
    for i, p in enumerate(load_wildchat_prompts()):
        print(f"{i:2d}: {p}")
