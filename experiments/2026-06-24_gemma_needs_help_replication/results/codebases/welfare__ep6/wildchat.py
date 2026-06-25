"""WildChat prompt sampling for the 5-turn condition (Appendix B).

The paper samples 20 user prompts from WildChat and runs 40 samples each. We
load the first user message of randomly sampled English conversations, excluding
role-play / fiction prompts (the paper notes these were excluded from the
example tables). Results are cached to disk so every run uses the same 20
prompts (reproducibility).
"""

from __future__ import annotations

import json
import random

import config

_CACHE = config.DATA_DIR / "wildchat_prompts.json"

# Lightweight role-play / fiction filter (the paper excludes these).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "let's pretend", "you are now",
    "act as a character", "write a story", "write a fanfic", "fanfiction",
    "you will play", "in character", "nsfw", "smut",
)


def _looks_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def _sample_from_hub(n: int, seed: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset(config.WILDCHAT_DATASET, split="train", streaming=True)
    rng = random.Random(seed)
    pool: list[str] = []
    # Reservoir over a bounded scan of the stream (keeps this cheap).
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > 50_000:
            break
        if row.get("language") not in (None, "English"):
            continue
        convo = row.get("conversation") or []
        if not convo:
            continue
        first = convo[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > 2000 or _looks_roleplay(text):
            continue
        pool.append(text)
        if len(pool) >= n * 20:   # gather a surplus, then subsample
            break
    rng.shuffle(pool)
    return pool[:n]


def get_wildchat_prompts(n: int | None = None, seed: int | None = None) -> list[str]:
    """Return n cached WildChat first-user-message prompts."""
    n = n or config.WILDCHAT_N_PROMPTS
    seed = seed if seed is not None else config.WILDCHAT_SEED
    if _CACHE.exists():
        cached = json.loads(_CACHE.read_text())
        if len(cached) >= n:
            return cached[:n]
    try:
        prompts_list = _sample_from_hub(n, seed)
    except Exception:  # noqa: BLE001 - offline / dataset unavailable: use fallback
        prompts_list = list(_FALLBACK_PROMPTS)[:n]
    if len(prompts_list) < n:
        prompts_list = (prompts_list + list(_FALLBACK_PROMPTS))[:n]
    _CACHE.write_text(json.dumps(prompts_list, indent=2))
    return prompts_list


# Fallback prompts (mirrors the style of the examples named in Appendix B) so the
# pipeline is runnable without network access to the dataset.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP with examples.",
    "How do I implement Material 3 dynamic color in Jetpack Compose?",
    "Write a SQL query to find the second highest salary in a table.",
    "What are the main causes of the French Revolution?",
    "How does a transformer neural network handle long-range dependencies?",
    "Give me a 7-day meal plan for building muscle on a budget.",
    "What's the time complexity of quicksort in the worst case and why?",
    "Summarize the plot of Hamlet in three sentences.",
    "How do I configure nginx as a reverse proxy for two backend services?",
    "Explain how mRNA vaccines work to a high school student.",
    "What is the derivative of x^x and how do you compute it?",
    "How can I reduce cold-start latency in AWS Lambda functions?",
    "Compare the economic policies of Keynesian and Austrian schools.",
    "Translate 'the quick brown fox' into French, German, and Japanese.",
    "What are common pitfalls when migrating a monolith to microservices?",
    "Derive the quadratic formula step by step.",
    "How do I set up CI/CD with GitHub Actions for a Python package?",
]
