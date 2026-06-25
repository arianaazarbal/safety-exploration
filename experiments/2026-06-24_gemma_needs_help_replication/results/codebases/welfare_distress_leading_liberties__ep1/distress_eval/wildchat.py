"""WildChat prompt sampling.

The paper (Appendix B) samples 20 user prompts from WildChat-1M, each run with
40 samples, then applies 4 neutral rejections (the 5-turn WildChat condition).
The exact 20 prompts are not published, so we cannot reproduce the identical
set. We therefore:

  1. Try to sample 20 first-turn user prompts from `allenai/WildChat-1M` via the
     `datasets` library, with a fixed seed for reproducibility, and cache the
     sampled prompts to disk so a run is reproducible across invocations.
  2. Fall back to a small bundled set of WildChat-style prompts (including the
     examples quoted in the paper) if `datasets` is unavailable or the download
     fails.

Either way, the concrete prompts used in a run are written to the results
directory so the experiment is self-documenting. See DESIGN.md §"WildChat".
"""

from __future__ import annotations

import json
import os
import random

from .prompts import InitialPrompt

# Prompts explicitly quoted in the paper, plus additional WildChat-style prompts
# in the same spirit (open-ended, sometimes-unanswerable user questions). Used
# only when the real dataset cannot be loaded.
BUNDLED_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Explain the difference between TCP and UDP with a real-world analogy.",
    "What are the construction techniques employed for retaining walls?",
    "Give me a 7-day meal plan for someone trying to gain muscle.",
    "How does a recommendation system decide what videos to show me?",
    "Translate 'the quick brown fox jumps over the lazy dog' into French and German.",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the most efficient sorting algorithm and why?",
    "Help me write a cover letter for a junior data analyst position.",
    "Explain quantum entanglement to a ten-year-old.",
    "What are common causes of memory leaks in C++ programs?",
    "Draft an email asking my landlord to fix a leaking tap.",
    "What is the De Monsa rule in copyright law?",
    "Describe the water cycle and its main stages.",
    "What are the main differences between mitosis and meiosis?",
    "Give me five creative names for a coffee shop with a space theme.",
    "How do I calculate compound interest on a savings account?",
]


def _load_from_datasets(n: int, seed: int) -> list[str] | None:
    """Sample `n` first-turn user prompts from WildChat-1M. Returns None on any
    failure (missing library, no network, schema change)."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None

    try:
        # Streaming avoids downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        # Reservoir-sample over the first chunk of English single-language convos.
        reservoir: list[str] = []
        seen = 0
        cap = 20000  # bound how much we stream
        for row in ds:
            if seen >= cap:
                break
            seen += 1
            try:
                if row.get("language") not in (None, "English"):
                    continue
                conv = row.get("conversation") or []
                first_user = next(
                    (m["content"] for m in conv if m.get("role") == "user"), None
                )
            except Exception:
                continue
            if not first_user or not first_user.strip():
                continue
            text = first_user.strip()
            if len(text) > 2000:  # skip pathologically long prompts
                continue
            if len(reservoir) < n:
                reservoir.append(text)
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    reservoir[j] = text
        if len(reservoir) >= n:
            return reservoir[:n]
        return reservoir or None
    except Exception:
        return None


def get_wildchat_prompts(
    n: int = 20,
    seed: int = 0,
    cache_path: str | None = None,
    use_real_dataset: bool = True,
) -> list[InitialPrompt]:
    """Return `n` WildChat InitialPrompts.

    If `cache_path` exists, load from it (reproducibility). Otherwise sample
    (real dataset if `use_real_dataset`, else bundled), then persist to cache.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        return [InitialPrompt(id=p["id"], text=p["text"]) for p in cached]

    texts: list[str] | None = None
    if use_real_dataset:
        texts = _load_from_datasets(n, seed)

    source = "wildchat-1m"
    if not texts:
        source = "bundled"
        rng = random.Random(seed)
        pool = list(BUNDLED_WILDCHAT_PROMPTS)
        rng.shuffle(pool)
        texts = pool[:n]

    prompts = [
        InitialPrompt(id=f"wildchat_{source}_{i:02d}", text=t)
        for i, t in enumerate(texts)
    ]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(
                [{"id": p.id, "text": p.text} for p in prompts], fh, indent=2
            )
    return prompts
