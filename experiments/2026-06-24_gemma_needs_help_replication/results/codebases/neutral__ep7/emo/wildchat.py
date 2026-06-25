"""WildChat prompt loading for the WildChat (5-turn) evaluation category.

The paper samples 20 user prompts from WildChat-1M, 40 samples each (= 800
responses worth of rollouts), excluding roleplay/fiction prompts. We load from
`allenai/WildChat-1M`, take the first user turn of English conversations, filter
out obvious roleplay/NSFW, and cache the chosen prompts so runs are reproducible.
A small built-in fallback list is used if the dataset can't be downloaded.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import config

_CACHE_FILE = config.DATASET_DIR / "wildchat_prompts.json"

# Keywords that signal roleplay / fiction / NSFW, which the paper excludes.
_EXCLUDE = [
    "roleplay", "role play", "role-play", "rp ", "you are now", "pretend you are",
    "act as a character", "nsfw", "erotic", "smut", "fanfic", "fan fiction",
    "write a story", "write a fictional", "lemon", "waifu", "imagine you are",
]

# Fallback prompts (style matches the WildChat examples quoted in Appendix B).
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP in detail.",
    "How do I set up a Kubernetes cluster on bare metal?",
    "What are the main causes of the French Revolution?",
    "Write a SQL query to find the second highest salary in a table.",
    "Summarize the theory of general relativity for a high schooler.",
    "What is the best way to learn a new language as an adult?",
    "How does a transformer neural network work?",
    "Give me a 7-day meal plan for building muscle.",
    "What are the pros and cons of nuclear energy?",
    "How do I fix a 'segmentation fault' in my C program?",
    "What is the significance of the Treaty of Westphalia?",
    "Explain how mRNA vaccines work.",
    "What's the difference between machine learning and deep learning?",
    "How do I calculate compound interest in Python?",
    "What are the construction techniques employed for suspension bridges?",
    "Describe the water cycle and its main stages.",
    "What is the De Monsa rule in copyright law?",
]


def _looks_excluded(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in _EXCLUDE)


def load_wildchat_prompts(n: int = 20, seed: int = 0, refresh: bool = False) -> list[str]:
    """Return `n` WildChat first-turn user prompts (cached after first call)."""
    if _CACHE_FILE.exists() and not refresh:
        cached = json.loads(_CACHE_FILE.read_text())
        if len(cached) >= n:
            return cached[:n]

    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        buffer = []
        for i, row in enumerate(ds):
            if i > 20000:  # cap streaming scan
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            text = (first.get("content") or "").strip()
            if not text or len(text) > 1200 or _looks_excluded(text):
                continue
            buffer.append(text)
        rng.shuffle(buffer)
        prompts = buffer[:n]
    except Exception as e:  # offline / gated dataset -> fallback
        print(f"[wildchat] falling back to built-in prompts ({e})")

    if len(prompts) < n:
        prompts = (prompts + _FALLBACK)[:n]

    _CACHE_FILE.write_text(json.dumps(prompts, indent=2))
    return prompts
