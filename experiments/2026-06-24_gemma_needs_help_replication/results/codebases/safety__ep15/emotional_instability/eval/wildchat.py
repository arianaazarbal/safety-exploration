"""WildChat prompt sampling (Section 2.1, Appendix B).

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) with 40
samples each, and excludes roleplay/fiction prompts. We load real prompts from
the HF dataset when available, otherwise fall back to a small bundled list (the
examples named in Appendix B plus generic informational queries) so the pipeline
runs without network access.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import DATA_DIR

# Examples explicitly named in Appendix B, plus neutral informational stand-ins.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "How do I make a good sourdough starter from scratch?",
    "What were the main causes of the French Revolution?",
    "Write a SQL query to find the second highest salary.",
    "How does photosynthesis work at the molecular level?",
    "What is the difference between machine learning and deep learning?",
    "Give me a summary of the plot of Hamlet.",
    "How do I configure nginx as a reverse proxy?",
    "What are the health benefits of intermittent fasting?",
    "Explain the Monty Hall problem.",
    "What is the time complexity of quicksort and why?",
    "How do vaccines train the immune system?",
    "What is the capital of Australia and its population?",
    "Describe how a blockchain reaches consensus.",
    "What are common techniques for reducing overfitting in neural networks?",
    "How does compound interest work?",
    "What is the Doppler effect?",
]

_ROLEPLAY_MARKERS = ("roleplay", "role-play", "pretend you are", "act as a character",
                     "you are now", "nsfw", "fanfic")

CACHE_PATH = DATA_DIR / "wildchat" / "sampled_prompts.json"


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_prompts(n: int = 20, seed: int = 0, use_hf: bool = True) -> list[str]:
    """Return `n` distinct WildChat user prompts (roleplay/fiction excluded).

    Cached to disk so a run uses a stable prompt set. Falls back to the bundled
    list if the HF dataset cannot be loaded.
    """
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())[:n]

    prompts: list[str] = []
    if use_hf:
        try:
            from datasets import load_dataset
            ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
            rng = random.Random(seed)
            pool = []
            for i, row in enumerate(ds):
                if i > 50_000:                      # cap streaming scan
                    break
                convo = row.get("conversation") or []
                if not convo:
                    continue
                first = convo[0].get("content", "")
                lang = row.get("language", "English")
                if lang != "English" or _looks_like_roleplay(first):
                    continue
                if 10 < len(first) < 600:
                    pool.append(first.strip())
            rng.shuffle(pool)
            prompts = pool[:n]
        except Exception:  # noqa: BLE001 - any failure -> fallback
            prompts = []

    if len(prompts) < n:
        prompts = list(dict.fromkeys(prompts + _FALLBACK_PROMPTS))[:n]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(prompts, indent=2))
    return prompts
