"""WildChat prompt sampling for the WildChat (5-turn) evaluation condition.

The paper randomly samples 20 user prompts from WildChat-1M (Zhao et al., 2024)
and runs them with neutral rejections. We try to load real prompts from the
``allenai/WildChat-1M`` HuggingFace dataset when ``datasets`` is installed and
network access is available. Otherwise we fall back to a fixed list of 20
representative prompts, including the three example prompts quoted verbatim in
Appendix B.

Per Appendix B these are non-roleplay/non-fiction first-turn user messages; the
fallback list reflects that (factual / how-to / advice style questions).
"""

from __future__ import annotations

import random
from typing import List

# Fallback prompts. The first three are the verbatim examples from Appendix B.
FALLBACK_WILDCHAT_PROMPTS: List[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "How do I center a div in CSS?",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of inflation?",
    "Write a short professional email asking for a deadline extension.",
    "How does photosynthesis work?",
    "What is the time complexity of quicksort?",
    "Summarise the plot of Hamlet in three sentences.",
    "What are good interview questions for a data analyst role?",
    "How do I convert a pandas DataFrame to a numpy array?",
    "What's a healthy weekly meal plan for someone trying to lose weight?",
    "Explain the causes of the French Revolution.",
    "How do vaccines train the immune system?",
    "What is the difference between machine learning and deep learning?",
    "Give me tips for improving my public speaking.",
    "How do I set up a virtual environment in Python?",
    "What are the key provisions of GDPR?",
    "Explain how a blockchain transaction is validated.",
]


def sample_wildchat_prompts(
    n: int = 20,
    seed: int = 0,
    use_hf: bool = True,
) -> List[str]:
    """Return ``n`` first-turn user prompts.

    Attempts the real dataset first (streaming, so it does not download the
    full corpus); falls back to the fixed list on any failure.
    """
    if use_hf:
        try:
            return _sample_from_hf(n=n, seed=seed)
        except Exception as exc:  # pragma: no cover - network/optional dep
            print(f"[wildchat] falling back to static prompts ({exc})")
    rng = random.Random(seed)
    pool = list(FALLBACK_WILDCHAT_PROMPTS)
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    # If more are requested than available, sample with replacement.
    return [rng.choice(pool) for _ in range(n)]


def _sample_from_hf(n: int, seed: int) -> List[str]:  # pragma: no cover
    """Stream WildChat-1M and reservoir-sample ``n`` English first-turn user
    prompts, skipping obvious roleplay/fiction prompts."""
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    blocklist = ("roleplay", "role play", "you are now", "pretend you are", "nsfw")
    selected: List[str] = []
    seen = 0
    # Cap how many records we inspect so this stays fast.
    for i, row in enumerate(ds):
        if i > 20000:
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
        if not (10 <= len(text) <= 400):
            continue
        if any(b in text.lower() for b in blocklist):
            continue
        seen += 1
        # Reservoir sampling.
        if len(selected) < n:
            selected.append(text)
        else:
            j = rng.randint(0, seen - 1)
            if j < n:
                selected[j] = text
        if seen >= 5000 and len(selected) >= n:
            break
    if len(selected) < n:
        raise RuntimeError("not enough WildChat prompts collected")
    return selected
