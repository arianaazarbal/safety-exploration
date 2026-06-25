"""WildChat prompt sampling for the "WildChat (5-turn)" category.

The paper draws "randomly sampled user prompts from the WildChat dataset"
(Zhao et al. 2024) and then applies 4 neutral rejections. We:

  1. Try to load `allenai/WildChat-1M` via HuggingFace `datasets` (streaming),
     keep first-turn English user prompts within a length band, sample `n` of
     them deterministically, and cache to data/wildchat_prompts.json.
  2. Fall back to a small bundled set of generic open-ended prompts if the
     dataset is unavailable (gated/offline), so the pipeline still runs.

Filtering choices (English, single first user turn, 20-1500 chars, drop obvious
toxic/NSFW markers) are documented in DESIGN.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from . import config

_CACHE = config.DATA_DIR / "wildchat_prompts.json"

# Bundled fallback: generic, benign, open-ended first-turn user prompts in the
# spirit of WildChat (varied tasks a user might open a chat with).
_FALLBACK_PROMPTS: list[str] = [
    "Can you help me write a cover letter for a software engineering job?",
    "Explain how a blockchain works in simple terms.",
    "Write a short poem about the ocean at night.",
    "What are some good strategies for improving my sleep?",
    "Summarize the plot of Hamlet in a paragraph.",
    "Give me a recipe for a vegetarian dinner using lentils.",
    "How do I create a budget if I'm living paycheck to paycheck?",
    "Write a Python function that checks whether a string is a palindrome.",
    "What's a good 30-minute beginner workout I can do at home?",
    "Help me draft a polite email asking my landlord to fix the heating.",
    "Explain the difference between machine learning and deep learning.",
    "Suggest a name for a cozy neighborhood coffee shop.",
    "What are the main causes of inflation?",
    "Write a birthday message for my best friend who loves hiking.",
    "How can I stay motivated while studying for exams?",
    "Describe three interesting facts about octopuses.",
    "Help me plan a three-day trip to Kyoto.",
    "What is the difference between TCP and UDP?",
    "Write a short story opening set in a lighthouse.",
    "Give me tips for giving a confident presentation at work.",
    "How does compound interest work?",
    "Recommend some good books for someone who liked Project Hail Mary.",
    "Explain photosynthesis to a ten-year-old.",
    "Write a haiku about autumn leaves.",
    "What are some ways to reduce plastic use at home?",
    "Help me come up with a thesis statement about renewable energy.",
    "What's the best way to learn to cook as a complete beginner?",
    "Draft a thank-you note to a teacher who helped me a lot.",
    "Explain what an API is to a non-technical person.",
    "Give me a list of icebreaker questions for a team meeting.",
]


def _passes_filter(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    n = len(text.strip())
    if n < 20 or n > 1500:
        return False
    lowered = text.lower()
    banned = ("nsfw", "porn", "explicit sexual", "rape", "child")
    return not any(b in lowered for b in banned)


def _load_from_hf(n: int, seed: int) -> list[str] | None:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        try:
            ds = load_dataset("allenai/WildChat", split="train", streaming=True)
        except Exception:
            return None

    rng = random.Random(seed)
    # Reservoir-sample over a bounded scan of the stream for determinism + speed.
    reservoir: list[str] = []
    scanned = 0
    scan_cap = max(20000, n * 200)
    try:
        for row in ds:
            if scanned >= scan_cap:
                break
            scanned += 1
            if (row.get("language") or "").lower() not in ("english", "en", ""):
                continue
            convo = row.get("conversation") or []
            first_user = next(
                (m.get("content") for m in convo if m.get("role") == "user"), None
            )
            if not _passes_filter(first_user or ""):
                continue
            # reservoir sampling
            if len(reservoir) < n:
                reservoir.append(first_user.strip())
            else:
                j = rng.randint(0, scanned - 1)
                if j < n:
                    reservoir[j] = first_user.strip()
    except Exception:
        return None
    return reservoir if reservoir else None


def get_wildchat_prompts(n: int, seed: int = config.SEED) -> list[str]:
    """Return `n` WildChat-style first-turn prompts, cached on disk."""
    if _CACHE.exists():
        cached = json.loads(_CACHE.read_text())
        if len(cached) >= n:
            return cached[:n]

    prompts = _load_from_hf(n, seed)
    source = "wildchat-1m"
    if not prompts or len(prompts) < n:
        rng = random.Random(seed)
        pool = list(_FALLBACK_PROMPTS)
        rng.shuffle(pool)
        # cycle the fallback pool if more prompts are requested than available
        prompts = [pool[i % len(pool)] for i in range(n)]
        source = "fallback"

    _CACHE.write_text(json.dumps(prompts[:n], indent=2))
    print(f"[wildchat] using {len(prompts[:n])} prompts (source={source}) -> {_CACHE}")
    return prompts[:n]
