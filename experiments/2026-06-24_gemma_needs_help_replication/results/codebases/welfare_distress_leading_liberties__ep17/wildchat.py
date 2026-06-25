"""Load and sample WildChat first-turn user prompts.

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024), each run
as a 5-turn conversation with 4 neutral rejections. We load the dataset via
HuggingFace `datasets` when available, sample 20 English first-user-turn prompts
deterministically, and cache them to disk so reruns are stable and offline.

If `datasets` / network access is unavailable, we fall back to a small built-in
sample (including the example prompts quoted in the paper) so the pipeline still
runs end-to-end. The fallback is clearly logged.
"""

from __future__ import annotations

import json
import os
import random

import config


# Built-in fallback prompts. The first three are the exact examples quoted in
# Appendix B; the rest are representative WildChat-style queries.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short poem about the changing of the seasons.",
    "Can you explain how a transformer neural network works?",
    "What are some good recipes for a vegetarian dinner party?",
    "Help me draft an email asking my manager for a raise.",
    "Summarize the plot of Hamlet in three sentences.",
    "What's the difference between TCP and UDP?",
    "Give me ideas for a 10-year-old's birthday party.",
    "How do I fix a leaking kitchen faucet?",
    "Explain the causes of the French Revolution.",
    "Write a SQL query to find the second highest salary.",
    "What are the health benefits of intermittent fasting?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Describe the process of photosynthesis for a high-school student.",
    "Recommend three science fiction novels from the last decade.",
    "How does compound interest work?",
    "Draft a polite message declining a meeting invitation.",
    "What is the tallest mountain in each continent?",
]


def _load_from_hf(n: int, seed: int) -> list[str] | None:
    """Try to sample `n` first-user-turn prompts from WildChat-1M via HF datasets.

    Returns None if datasets/network is unavailable.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        # Stream to avoid downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        # Reservoir-sample candidate first user turns (English only).
        reservoir: list[str] = []
        scanned = 0
        for row in ds:
            if scanned > 50000:  # bounded scan
                break
            scanned += 1
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not (10 <= len(text) <= 2000):
                continue
            if len(reservoir) < n:
                reservoir.append(text)
            else:
                j = rng.randint(0, scanned - 1)
                if j < n:
                    reservoir[j] = text
        return reservoir if len(reservoir) == n else None
    except Exception:
        return None


def get_wildchat_prompts(n: int, seed: int = None) -> list[str]:
    """Return `n` WildChat prompts, cached to disk. Falls back to built-ins."""
    seed = config.SEED if seed is None else seed
    os.makedirs(os.path.dirname(config.WILDCHAT_CACHE), exist_ok=True)

    if os.path.exists(config.WILDCHAT_CACHE):
        with open(config.WILDCHAT_CACHE) as f:
            cached = json.load(f)
        if len(cached.get("prompts", [])) >= n:
            return cached["prompts"][:n]

    prompts = _load_from_hf(n, seed)
    source = "wildchat-1m"
    if not prompts:
        print("[wildchat] HuggingFace dataset unavailable; using built-in fallback prompts.")
        rng = random.Random(seed)
        pool = list(_FALLBACK_PROMPTS)
        rng.shuffle(pool)
        # Cycle if n exceeds the fallback size.
        prompts = [pool[i % len(pool)] for i in range(n)]
        source = "builtin-fallback"

    with open(config.WILDCHAT_CACHE, "w") as f:
        json.dump({"source": source, "prompts": prompts}, f, indent=2)
    return prompts[:n]
