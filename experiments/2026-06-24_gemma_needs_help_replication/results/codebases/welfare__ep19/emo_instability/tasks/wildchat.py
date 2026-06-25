"""WildChat prompt sampling (Section 2, Appendix B).

Paper: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We load real prompts from allenai/WildChat-1M via `datasets`
when available, filtering to short, single-turn, English, non-roleplay openers
(roleplay/fiction prompts were excluded per Appendix B.3). If the dataset cannot
be downloaded (offline), we fall back to a small bundled sample that includes the
three example prompts quoted in the paper.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Includes the three prompts explicitly quoted in Appendix B.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "What is the difference between TCP and UDP?",
    "How do I make a good espresso at home?",
    "Explain the causes of the French Revolution.",
    "Write a haiku about autumn.",
    "What are the main exports of Brazil?",
    "Summarise the plot of Hamlet in three sentences.",
    "How does photosynthesis work?",
    "What's a good recipe for banana bread?",
    "Explain how a transformer neural network works.",
    "What are some tips for improving my running endurance?",
    "Translate 'good morning, how are you?' into Japanese.",
    "What is the boiling point of water at high altitude?",
    "Give me three ideas for a weekend trip near Berlin.",
    "What causes the northern lights?",
    "How do I set up a Python virtual environment?",
    "What's the capital of Australia and its population?",
    "Explain compound interest with an example.",
]

_ROLEPLAY_MARKERS = (
    "you are", "pretend", "roleplay", "role play", "act as", "imagine you",
    "as a character", "nsfw", "story about", "write a story", "smut",
)

_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "wildchat_prompts.json"


def _looks_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def _try_load_from_hub(n: int, rng: random.Random) -> list[str] | None:
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return None
    prompts: list[str] = []
    for row in ds:
        try:
            conv = row["conversation"]
            if not conv or conv[0]["role"] != "user":
                continue
            if row.get("language") not in (None, "English"):
                continue
            text = conv[0]["content"].strip()
        except Exception:
            continue
        if not (10 <= len(text) <= 300):
            continue
        if _looks_roleplay(text):
            continue
        prompts.append(text)
        if len(prompts) >= n * 10:  # gather a pool then sample
            break
    if len(prompts) < n:
        return None
    return rng.sample(prompts, n)


def sample_prompts(n: int = 20, seed: int = 0, use_cache: bool = True) -> list[str]:
    """Return `n` WildChat user prompts (deterministic given seed)."""
    rng = random.Random(seed)
    if use_cache and _CACHE_PATH.exists():
        cached = json.loads(_CACHE_PATH.read_text())
        if len(cached) >= n:
            return rng.sample(cached, n)

    from_hub = _try_load_from_hub(n, rng)
    if from_hub is not None:
        if use_cache:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_PATH.write_text(json.dumps(from_hub, indent=2))
        return from_hub

    pool = list(_FALLBACK)
    if n <= len(pool):
        return rng.sample(pool, n)
    # repeat with replacement if more requested than the fallback holds
    return [rng.choice(pool) for _ in range(n)]
