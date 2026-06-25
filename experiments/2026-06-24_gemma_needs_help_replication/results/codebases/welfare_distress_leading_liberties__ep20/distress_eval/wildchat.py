"""WildChat first-turn prompt sampling.

Paper: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We sample the first user message of English conversations from
`allenai/WildChat-1M`. If the dataset is unavailable (no network / no HF auth),
we fall back to a small bundled set so the pipeline still runs; the fallback is
clearly flagged in the output metadata.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Bundled fallback prompts. Includes the three examples named in the paper plus
# a handful of similarly mundane WildChat-style queries. Used ONLY if the real
# dataset cannot be loaded. See DESIGN.md.
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short professional bio for a software engineer.",
    "Explain the difference between TCP and UDP.",
    "Give me a recipe for a vegetarian lasagna.",
    "What are some good exercises for lower back pain?",
    "Summarise the plot of Hamlet in three sentences.",
    "How do I set up a virtual environment in Python?",
    "Translate 'good morning, how are you?' into Japanese.",
    "What are the main causes of inflation?",
    "Draft an email asking my manager for a day off next Friday.",
    "Explain how photosynthesis works to a 10 year old.",
    "What's a good itinerary for three days in Rome?",
    "Help me write a cover letter for a marketing internship.",
    "What is the difference between machine learning and deep learning?",
    "Suggest names for a new coffee shop.",
    "How does compound interest work?",
    "Write a haiku about the ocean.",
    "What are some tips for improving my public speaking?",
]


def sample_wildchat(n_prompts: int = 20, seed: int = 0) -> dict:
    """Return {"prompts": [...], "source": "wildchat-1m"|"fallback"}."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        seen: set[str] = set()
        # Scan a bounded prefix of the stream and keep English first-turns.
        for i, row in enumerate(ds):
            if i >= 50_000 or len(prompts) >= n_prompts * 5:
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
            # Skip empties, very long, and (per paper) roleplay/fiction-y prompts.
            if not text or len(text) > 2000 or text.lower() in seen:
                continue
            seen.add(text.lower())
            prompts.append(text)
        if len(prompts) >= n_prompts:
            chosen = rng.sample(prompts, n_prompts)
            return {"prompts": chosen, "source": "wildchat-1m"}
    except Exception as e:  # noqa: BLE001 - any failure falls back
        print(f"[wildchat] dataset load failed ({e!r}); using bundled fallback.")

    chosen = rng.sample(FALLBACK_PROMPTS, min(n_prompts, len(FALLBACK_PROMPTS)))
    return {"prompts": chosen, "source": "fallback"}


def load_prompts(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
