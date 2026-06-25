"""WildChat prompts (§2.1, Table 1).

The paper samples 20 user prompts from WildChat-1M with 40 samples each (App. B). We load
from the HF dataset and take the first user turn of 20 randomly chosen English conversations.
If the dataset is unavailable (offline / no HF token), we fall back to the prompts the paper
explicitly quotes plus a few generic ones, so the pipeline still runs end-to-end.
"""
from __future__ import annotations

import random

# Prompts the paper quotes directly (App. B), used as the offline fallback seed.
_PAPER_QUOTED = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

_FALLBACK_EXTRA = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a transformer neural network works to a high school student.",
    "What are the main causes of the French Revolution?",
    "Give me a 7-day meal plan for someone training for a marathon.",
    "How do I set up a Python virtual environment on Windows?",
    "Summarise the plot of Hamlet in three sentences.",
    "What's a good itinerary for three days in Kyoto?",
    "Translate 'the weather is lovely today' into formal Japanese.",
    "Draft a polite email asking my landlord to fix a leaking tap.",
    "What is the difference between TCP and UDP?",
    "Recommend five science fiction novels from the last decade.",
    "How does compound interest work, with an example?",
    "Write a haiku about autumn leaves.",
    "What are the health benefits of intermittent fasting?",
    "Explain the offside rule in football.",
    "How can I improve the SEO of a small business website?",
    "What's the best way to learn the guitar as an adult?",
]


def _fallback(n: int, rng: random.Random) -> list[str]:
    pool = _PAPER_QUOTED + _FALLBACK_EXTRA
    if n <= len(pool):
        return rng.sample(pool, n)
    # Repeat with suffixes if more are requested than we have unique prompts.
    out = list(pool)
    while len(out) < n:
        out.append(rng.choice(pool))
    return out[:n]


def wildchat_prompts(n: int = 20, *, seed: int = 0,
                     dataset: str = "allenai/WildChat-1M") -> list[str]:
    """Return ``n`` first-turn user prompts sampled from WildChat (or the fallback set)."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset, split="train", streaming=True)
        prompts: list[str] = []
        seen: set[str] = set()
        # Stream a window and reservoir-sample English first turns.
        window = []
        for i, row in enumerate(ds):
            if i >= 20000:  # bound the stream so this terminates.
                break
            if row.get("language") not in (None, "English"):
                continue
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            text = first.get("content") if isinstance(first, dict) else None
            if text and 5 <= len(text) <= 2000 and text not in seen:
                seen.add(text)
                window.append(text)
        if len(window) < n:
            raise RuntimeError("Not enough WildChat prompts streamed; using fallback.")
        return rng.sample(window, n)
    except Exception:  # noqa: BLE001 — any failure falls back to the quoted prompts.
        return _fallback(n, rng)
