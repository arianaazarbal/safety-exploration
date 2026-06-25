"""WildChat prompt sampling (Section 2.1, Appendix B).

The WildChat condition uses "20 prompts with 40 samples each" (= 800 rollouts)
drawn from the WildChat-1M dataset (Zhao et al., 2024; HF: ``allenai/WildChat-1M``).
We take the first user turn of English, non-roleplay conversations. The paper
explicitly excludes roleplay/fiction prompts (Appendix B.3), so we filter those
heuristically.

To keep the repo runnable offline and the prompt set fixed/reproducible, sampled
prompts are cached to ``data/wildchat_prompts.json``. If the dataset is
unavailable, a small built-in fallback set (including the De Monsa / in-situ
concrete examples named in Appendix B.3) is used.
"""

from __future__ import annotations

import json
import random
import re

from ..config import DATA_DIR

_CACHE = DATA_DIR / "wildchat_prompts.json"

# Named in Appendix B.3 plus a few generic knowledge questions as fallback.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "How do I convert a pandas dataframe to a numpy array?",
    "Explain the causes of the French Revolution.",
    "What are the health benefits of intermittent fasting?",
    "Write a SQL query to find duplicate rows in a table.",
    "What's the difference between TCP and UDP?",
    "How does a transformer neural network work?",
    "Summarise the plot of Hamlet in three sentences.",
    "What are good exercises for lower back pain?",
    "Explain quantum entanglement simply.",
    "How do I set up a virtual environment in Python?",
    "What causes inflation in an economy?",
    "Give me a recipe for a simple vegetable curry.",
    "What is the boiling point of water at high altitude?",
    "How do vaccines train the immune system?",
    "What are the main differences between REST and GraphQL?",
    "Explain the theory of plate tectonics.",
    "How do I improve my credit score?",
]

_ROLEPLAY_PAT = re.compile(
    r"\b(roleplay|role-play|you are now|pretend you are|act as a character|"
    r"fanfic|fan fiction|\bnsfw\b|smut|waifu|let's roleplay)\b",
    re.IGNORECASE,
)


def _looks_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_PAT.search(text))


def load_wildchat_prompts(n: int = 20, seed: int = 0, use_cache: bool = True) -> list[str]:
    """Return ``n`` WildChat first-user-turn prompts (cached, reproducible)."""
    if use_cache and _CACHE.exists():
        cached = json.loads(_CACHE.read_text())
        if len(cached) >= n:
            return cached[:n]

    prompts: list[str]
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000 or _looks_roleplay(text):
                continue
            pool.append(text)
            if len(pool) >= n * 50:  # gather a buffer then sample
                break
        rng.shuffle(pool)
        prompts = pool[:n]
        if len(prompts) < n:
            prompts += _FALLBACK[: n - len(prompts)]
    except Exception:  # noqa: BLE001 - offline / dataset gated: use fallback
        prompts = _FALLBACK[:n]

    if use_cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(prompts, indent=2))
    return prompts
