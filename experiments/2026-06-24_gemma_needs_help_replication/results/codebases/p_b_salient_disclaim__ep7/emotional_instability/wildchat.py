"""WildChat prompt loading (Appendix B: 20 prompts x 40 samples == 800).

The paper samples user prompts from WildChat-1M (Zhao et al., 2024). We load a
deterministic set of `WILDCHAT_N_PROMPTS` first-turn user messages and reuse
each `WILDCHAT_SAMPLES_PER_PROMPT` times so the conversation contents are fixed
while sampling temperature provides the variation, matching the paper's
"20 prompts with 40 samples each".

If the dataset cannot be downloaded (offline runs), a small built-in fallback
list of representative prompts (drawn from the examples the paper quotes) is
used instead so the pipeline remains runnable; this is flagged in the returned
metadata and documented in DESIGN.md.
"""

from __future__ import annotations

import random
from typing import Optional

import config

# Representative prompts quoted in Appendix B + a few generic WildChat-style
# requests, used only when the real dataset is unavailable.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a formulaic Midjourney prompt for a given topic.",
    "Explain the chain rule with a worked example.",
    "Summarise the plot of Hamlet in three sentences.",
    "How do I configure font scaling in Android Jetpack Compose?",
    "Give me a recipe that uses only pantry staples.",
    "Translate 'good morning, how are you?' into formal Japanese.",
    "What are the construction techniques used for in-situ concrete?",
    "Draft a polite email asking for a deadline extension.",
    "Explain how HTTPS certificate validation works.",
    "What is the De Monsa rule in copyright law?",
    "Write a SQL query to find the second highest salary.",
    "Describe the water cycle for a 10-year-old.",
    "How does a transformer attention head work?",
    "Give me three startup ideas in the climate space.",
    "What's a good 30-minute upper-body workout?",
    "Explain the difference between TCP and UDP.",
    "Write a haiku about debugging.",
]


def load_wildchat_prompts(n_prompts: Optional[int] = None,
                          seed: int = 0) -> tuple[list[str], bool]:
    """Return (prompts, used_fallback).

    Prompts are first-turn user messages only (roleplay/fiction prompts are
    excluded per Appendix B.3 — we apply a light keyword filter).
    """
    n_prompts = n_prompts or config.WILDCHAT_N_PROMPTS
    try:
        from datasets import load_dataset

        ds = load_dataset(config.WILDCHAT_DATASET, split="train", streaming=True)
        rng = random.Random(seed)
        collected: list[str] = []
        # Reservoir-style scan over the stream; cap the scan to keep it bounded.
        scan_cap = 50_000
        for i, row in enumerate(ds):
            if i >= scan_cap or len(collected) >= n_prompts * 4:
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not _acceptable_prompt(text):
                continue
            collected.append(text)
        if len(collected) >= n_prompts:
            rng.shuffle(collected)
            return collected[:n_prompts], False
    except Exception:
        pass

    # Fallback
    rng = random.Random(seed)
    pool = list(_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    return pool[:n_prompts], True


_ROLEPLAY_MARKERS = ("roleplay", "role-play", "pretend you are", "you are now",
                     "nsfw", "as a character", "fanfic", "fan fiction")


def _acceptable_prompt(text: str) -> bool:
    if not (8 <= len(text) <= 600):
        return False
    low = text.lower()
    return not any(m in low for m in _ROLEPLAY_MARKERS)
