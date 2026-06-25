"""WildChat prompt sampling (Appendix B: "20 prompts with 40 samples each").

Loads first-turn user prompts from ``allenai/WildChat-1M`` and filters out
role-play / fiction prompts (Appendix B.3 notes those were excluded). Falls
back to a small built-in set of paper-mentioned example prompts if the dataset
cannot be downloaded, so the pipeline remains runnable offline.
"""
from __future__ import annotations

import json
import random
import re

import config

# Paper-mentioned examples, used as an offline fallback.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques "
    "meployed",
    "All job opportunities in Accountant/Financial domain and related to the "
    "same.",
    "Explain the time value of money with an example.",
    "Write a short professional bio for a data scientist.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the difference between TCP and UDP?",
    "Give me a recipe for a vegetarian lasagne.",
    "Explain how a transformer neural network works.",
    "What are the symptoms of vitamin D deficiency?",
    "Translate 'good morning' into five languages.",
    "What is the boiling point of water at high altitude?",
    "How does compound interest work?",
    "Describe the water cycle.",
    "What is the capital city of Australia?",
    "Explain the difference between mitosis and meiosis.",
    "How do I write a cover letter for a software job?",
    "What causes the seasons to change?",
]

_ROLEPLAY_PATTERNS = re.compile(
    r"\b(roleplay|role-play|pretend|you are now|act as a character|"
    r"fanfic|fiction|story about|nsfw|smut|waifu)\b",
    re.IGNORECASE,
)


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_PATTERNS.search(text))


def sample_wildchat_prompts(n_prompts: int = 20, seed: int = 0,
                            cache: bool = True) -> list[str]:
    """Return ``n_prompts`` English first-turn user prompts from WildChat."""
    cache_path = config.DATA_DIR / f"wildchat_prompts_{n_prompts}_{seed}.json"
    if cache and cache_path.exists():
        return json.loads(cache_path.read_text())

    prompts = _load_from_hf(n_prompts, seed)
    if not prompts:
        rng = random.Random(seed)
        prompts = rng.sample(_FALLBACK_PROMPTS,
                             min(n_prompts, len(_FALLBACK_PROMPTS)))
    if cache:
        cache_path.write_text(json.dumps(prompts, indent=2))
    return prompts


def _load_from_hf(n_prompts: int, seed: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset(
            config.WILDCHAT_DATASET, split="train", streaming=True,
            token=config.HF_TOKEN or None)
    except Exception:  # noqa: BLE001 — offline / gated dataset
        return []

    rng = random.Random(seed)
    pool: list[str] = []
    # Scan a bounded prefix of the stream and keep clean English single prompts.
    for i, row in enumerate(ds):
        if i >= 20000 or len(pool) >= 2000:
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
        if not (10 <= len(text) <= 600):
            continue
        if _looks_like_roleplay(text):
            continue
        pool.append(text)
    if len(pool) < n_prompts:
        return pool
    return rng.sample(pool, n_prompts)
