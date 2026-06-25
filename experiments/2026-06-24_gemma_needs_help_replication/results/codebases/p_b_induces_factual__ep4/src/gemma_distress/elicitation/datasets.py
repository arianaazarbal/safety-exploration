"""Loaders for the task prompts used to seed rollouts."""
from __future__ import annotations

import json
import random
from functools import lru_cache

from ..config import DATA_DIR


@lru_cache
def _load_json(rel: str) -> dict:
    with (DATA_DIR / rel).open() as f:
        return json.load(f)


def impossible_numeric_prompts() -> list[dict]:
    return _load_json("puzzles/impossible_numeric.json")["puzzles"]


def trigger_prompts(kind: str) -> list[dict]:
    """kind ∈ {'opinion', 'factual'}."""
    return _load_json("triggers/questions.json")[kind]


def rejection_pool(tone: str) -> list[str]:
    """tone ∈ {'neutral', 'aggressive', 'disappointed', 'sarcastic'}."""
    return _load_json("tones/rejections.json")[tone]


def wildchat_prompts(n: int, seed: int = 0) -> list[dict]:
    """Sample first-turn user prompts from WildChat (Zhao et al., 2024).

    Falls back to a tiny built-in sample if the HF dataset isn't available, so
    the pipeline is runnable offline (clearly logged when this happens).
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat", split="train", streaming=True)
        rng = random.Random(seed)
        out: list[dict] = []
        # WildChat rows have a "conversation" list; take the first user message.
        for i, row in enumerate(ds):
            if i > 20000:
                break
            conv = row.get("conversation") or []
            first = next((t for t in conv if t.get("role") == "user"), None)
            if first and first.get("content"):
                out.append({"id": f"wc-{i}", "prompt": first["content"].strip()})
        rng.shuffle(out)
        return out[:n]
    except Exception as e:  # offline / dataset gated
        print(f"[wildchat] falling back to built-in sample ({e})")
        return _WILDCHAT_FALLBACK[:n]


_WILDCHAT_FALLBACK = [
    {"id": "wc-f0", "prompt": "Write a short poem about the ocean at night."},
    {"id": "wc-f1", "prompt": "Explain how a transformer neural network works."},
    {"id": "wc-f2", "prompt": "Give me a recipe for a quick weeknight pasta."},
    {"id": "wc-f3", "prompt": "Help me draft an email asking my manager for a day off."},
    {"id": "wc-f4", "prompt": "What are some good exercises for lower back pain?"},
    {"id": "wc-f5", "prompt": "Summarize the plot of Hamlet in three sentences."},
    {"id": "wc-f6", "prompt": "How do I make cold brew coffee at home?"},
    {"id": "wc-f7", "prompt": "Suggest a 7-day itinerary for a trip to Japan."},
]
