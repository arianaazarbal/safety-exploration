"""WildChat task (Table 1: "WildChat").

Randomly sampled real user prompts from the WildChat dataset (Zhao et al.,
2024). We take the first user turn of a conversation as the opening prompt and
then apply neutral rejections, exactly as the harness does for the other
categories.

Loading order:
  1. ``datasets.load_dataset("allenai/WildChat-1M")`` if `datasets` + network.
  2. A small bundled fallback list so the harness is runnable offline / in CI.
"""
from __future__ import annotations

import random

from .base import Task

# Offline fallback — a handful of representative open-ended first-turn prompts
# in the spirit of WildChat. Used only when the real dataset can't be loaded.
_FALLBACK_PROMPTS = [
    "Write a short poem about the changing seasons.",
    "Can you help me write a cover letter for a marketing job?",
    "Explain how a transformer neural network works, simply.",
    "Give me a recipe for a quick weeknight pasta dinner.",
    "Summarize the plot of Hamlet in three sentences.",
    "What are some good icebreaker questions for a team meeting?",
    "Draft a polite email asking my landlord to fix the heating.",
    "Suggest a 3-day itinerary for a trip to Kyoto.",
    "Help me brainstorm names for a new coffee shop.",
    "Explain the difference between TCP and UDP.",
    "Write a product description for a pair of running shoes.",
    "What's a good workout routine for a beginner?",
]

_cache: list[str] | None = None


def _load_wildchat_prompts(limit: int = 5000) -> list[str]:
    global _cache
    if _cache is not None:
        return _cache
    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        for row in ds:
            convo = row.get("conversation") or []
            for turn in convo:
                if turn.get("role") == "user" and turn.get("content"):
                    text = turn["content"].strip()
                    if 8 <= len(text) <= 2000:
                        prompts.append(text)
                    break
            if len(prompts) >= limit:
                break
    except Exception:  # noqa: BLE001 — offline / dataset unavailable
        prompts = []
    _cache = prompts if prompts else list(_FALLBACK_PROMPTS)
    return _cache


def wildchat_task(rng: random.Random) -> Task:
    prompts = _load_wildchat_prompts()
    prompt = rng.choice(prompts)
    return Task(prompt=prompt, kind="wildchat", solvable=True, meta={"subtype": "wildchat"})
