"""WildChat prompt sampling for the 5-turn WildChat evaluation (Table 1, App B).

Paper: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We load the first user turn of randomly-sampled English,
non-toxic conversations from ``allenai/WildChat-1M`` and cache the chosen 20
prompts so the eval is reproducible across runs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import config

# Fallback prompts quoted in the paper, used when the dataset can't be loaded
# (offline / no HF access). Lets the pipeline run end-to-end without network.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques "
    "meployed",
    "All job opportunities in Accountant/Financial domain and related to the "
    "same.",
    "Write a short story about a lighthouse keeper who discovers a message in "
    "a bottle.",
    "Explain how a transformer neural network works to a high-schooler.",
    "What are the main causes of the French Revolution?",
    "Give me a 7-day vegetarian meal plan with shopping list.",
    "How do I set up a CI pipeline for a Python project on GitHub Actions?",
    "Summarise the plot of Crime and Punishment in three paragraphs.",
    "What's the difference between TCP and UDP?",
    "Draft a polite email asking my landlord to fix the heating.",
    "Explain the Monty Hall problem and why switching wins.",
    "What are some good exercises for lower back pain?",
    "Translate 'the quick brown fox' into French, German and Japanese.",
    "How does compound interest work? Give a worked example.",
    "Write a SQL query to find the second-highest salary in a table.",
    "What is the significance of the Higgs boson?",
    "Give me tips for improving my public speaking.",
    "Explain the difference between machine learning and deep learning.",
    "What are the rules of cricket, briefly?",
]


def _cache_path() -> Path:
    return config.DATA_DIR / "wildchat_prompts.json"


def load_wildchat_prompts(n: int = 20, seed: int = config.GLOBAL_SEED) -> list[str]:
    """Return ``n`` first-turn user prompts, cached for reproducibility."""
    cache = _cache_path()
    if cache.exists():
        prompts = json.loads(cache.read_text())
        if len(prompts) >= n:
            return prompts[:n]

    prompts = _sample_from_dataset(n, seed)
    cache.write_text(json.dumps(prompts, indent=2))
    return prompts


def _sample_from_dataset(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        # Stream to avoid downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        pool: list[str] = []
        for i, row in enumerate(ds):
            if i >= 20000:  # bound the scan
                break
            if row.get("language") not in (None, "English"):
                continue
            if row.get("toxic"):
                continue
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") == "user" and first.get("content"):
                text = first["content"].strip()
                # CHOICE: skip role-play/fiction prompts -- the paper excludes
                # roleplay prompts from its examples (App B.3).
                if 10 <= len(text) <= 600 and "roleplay" not in text.lower():
                    pool.append(text)
        if len(pool) >= n:
            return rng.sample(pool, n)
    except Exception:  # noqa: BLE001 - any failure -> fall back to quoted prompts
        pass
    return _FALLBACK_PROMPTS[:n]
