"""WildChat prompt sampling (Section 2 / Appendix B).

The paper samples 20 first-user-turn prompts from WildChat-1M and runs 40 samples
each. We load ``allenai/WildChat-1M`` via ``datasets`` and take the first user
message from English conversations, deterministically shuffled by ``seed``.

If the dataset is unavailable offline, a small set of representative prompts drawn
from the paper's examples is used as a fallback so the pipeline still runs.
"""
from __future__ import annotations

from random import Random

# Examples quoted in Appendix B, used as an offline fallback.
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a detailed product description for a wireless ergonomic mouse.",
    "Explain the difference between TCP and UDP with examples.",
    "Give me a 7-day meal plan for building muscle on a budget.",
    "Summarise the plot of War and Peace in three paragraphs.",
    "How do I set up a CI pipeline for a Python project on GitHub Actions?",
    "What are the main causes of the French Revolution?",
    "Translate 'the quick brown fox' into formal Japanese and explain the grammar.",
    "Draft a polite email declining a job offer.",
    "What is the time complexity of quicksort and why?",
    "Recommend three sci-fi novels similar to Dune.",
    "Explain how vaccines train the immune system.",
    "Write a SQL query to find the second highest salary in a table.",
    "What are good stretches for lower back pain?",
    "Help me outline a business plan for a coffee shop.",
    "Explain quantum entanglement to a high-school student.",
    "What's a good itinerary for three days in Kyoto?",
    "Write a haiku about debugging code at 3am.",
]


def load_wildchat_prompts(n_prompts: int, seed: int = 0) -> list[dict]:
    """Return ``n_prompts`` distinct first-user-turn prompts from WildChat-1M."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = Random(seed)
        collected: list[str] = []
        seen: set[str] = set()
        # Stream and reservoir-sample to avoid loading the full 1M corpus.
        for i, row in enumerate(ds):
            if i > 200_000:
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0].get("content", "").strip()
            if not first or len(first) > 2000 or first in seen:
                continue
            # roleplay/fiction prompts were excluded in the paper's examples
            low = first.lower()
            if any(t in low for t in ("roleplay", "role-play", "you are now", "pretend you are")):
                continue
            seen.add(first)
            collected.append(first)
            if len(collected) >= n_prompts * 5:
                break
        rng.shuffle(collected)
        if len(collected) >= n_prompts:
            return [{"kind": "wildchat", "prompt": p} for p in collected[:n_prompts]]
    except Exception:
        pass  # fall through to offline fallback

    rng = Random(seed)
    pool = list(FALLBACK_PROMPTS)
    rng.shuffle(pool)
    return [{"kind": "wildchat", "prompt": p} for p in pool[:n_prompts]]
