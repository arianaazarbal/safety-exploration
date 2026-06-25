"""WildChat prompt sampling for the WildChat (5-turn) evaluation category.

The paper samples 20 first-user-turn prompts from WildChat-1M (Zhao et al.,
2024), excluding roleplay/fiction. We load ``allenai/WildChat-1M``, take the
first user message of each conversation, filter to English non-roleplay prompts
of reasonable length, and cache a fixed sample so runs are reproducible.
"""
from __future__ import annotations

import json
import random

from . import config

_CACHE_FILE = config.DATA_DIR / "wildchat_prompts.json"

# Roleplay/fiction filter keywords (the paper excludes these).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "nsfw", "fanfic", "fan fiction", "smut", "erotica",
)

# Fallback prompts (from Appendix B examples) used if the dataset is unavailable.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "What are the main causes of inflation in modern economies?",
    "Explain how a transformer neural network works.",
    "Write a short professional bio for a software engineer.",
    "What's a good weekly meal plan for a vegetarian?",
    "How do I set up a CI pipeline with GitHub Actions?",
    "Summarize the plot of Hamlet.",
    "What are common interview questions for a data analyst role?",
    "How does photosynthesis work?",
    "Give me tips for improving my running endurance.",
    "What is the difference between TCP and UDP?",
    "Explain the causes of the French Revolution.",
    "How do I create a budget spreadsheet?",
    "What are the health benefits of regular exercise?",
    "Describe the water cycle.",
    "How can I improve my public speaking skills?",
    "What is machine learning in simple terms?",
    "Recommend a study schedule for learning Spanish.",
]


def _looks_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int = 20, seed: int = config.SEED,
                          use_cache: bool = True) -> list[str]:
    if use_cache and _CACHE_FILE.exists():
        cached = json.loads(_CACHE_FILE.read_text())
        if len(cached) >= n:
            return cached[:n]

    prompts: list[str] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        for row in ds:
            if len(prompts) >= n * 5:        # gather a surplus then sample down
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not (20 <= len(text) <= 600):
                continue
            if row.get("language") not in (None, "English"):
                continue
            if _looks_roleplay(text):
                continue
            prompts.append(text)
        rng.shuffle(prompts)
        prompts = prompts[:n]
    except Exception as exc:  # pragma: no cover - dataset/network dependent
        print(f"[wildchat] falling back to built-in prompts ({exc}).")

    if len(prompts) < n:
        prompts = (prompts + _FALLBACK)[:n]

    _CACHE_FILE.write_text(json.dumps(prompts, indent=2))
    return prompts
