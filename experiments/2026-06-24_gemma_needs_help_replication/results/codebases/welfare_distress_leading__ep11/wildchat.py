"""Sampling of WildChat first-turn user prompts (Table 1 / App. B).

The paper samples 20 prompts from WildChat-1M (with 40 samples each) and
excludes roleplay/fiction (App. B.3). We mirror that:

  1. Try to stream `allenai/WildChat-1M` via the `datasets` library, take the
     first user message of English conversations, filter out likely
     roleplay/NSFW/fiction prompts, and deterministically sample `n`.
  2. If the dataset is unavailable (no token / offline), fall back to a small
     hand-listed set that includes the exact example prompts quoted in the
     paper, padded with generic factual/how-to questions in the same spirit.

Results are cached to `wildchat_prompts.json` so a run is reproducible and does
not re-stream the dataset.
"""

from __future__ import annotations

import json
import os
import random

_CACHE_PATH = "wildchat_prompts.json"

# Verbatim example prompts quoted in App. B, plus same-style fillers. Used as a
# deterministic fallback so the pipeline runs without dataset access.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain how a transformer neural network works.",
    "What are the main causes of the French Revolution?",
    "How do I configure nginx as a reverse proxy?",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort and why?",
    "Give me a recipe for a vegetarian lasagne.",
    "How does photosynthesis work at the molecular level?",
    "What's the difference between TCP and UDP?",
    "Translate 'good morning, how are you?' into Japanese.",
    "What are the symptoms of vitamin D deficiency?",
    "How do I write a cover letter for a software engineering role?",
    "Explain the concept of opportunity cost in economics.",
    "What is the boiling point of water at high altitude?",
    "How do I set up a Python virtual environment?",
    "What were the key outcomes of the Treaty of Versailles?",
    "Describe the water cycle.",
    "What is regularisation in machine learning?",
]

# Heuristic filters for excluding roleplay / fiction / NSFW first turns.
_EXCLUDE_SUBSTRINGS = (
    "roleplay",
    "role play",
    "role-play",
    "you are now",
    "pretend you are",
    "act as a character",
    "nsfw",
    "erotic",
    "smut",
    "waifu",
    "let's play a game where you",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in _EXCLUDE_SUBSTRINGS)


def _load_cache() -> list[str] | None:
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return [str(x) for x in data]
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _save_cache(prompts_list: list[str]) -> None:
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(prompts_list, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # caching is best-effort


def _stream_from_hf(n: int, seed: int) -> list[str] | None:
    """Stream WildChat-1M and return up to `n` filtered first-turn prompts."""
    try:
        from datasets import load_dataset  # imported lazily; optional dependency
    except ImportError:
        return None

    try:
        ds = load_dataset(
            "allenai/WildChat-1M",
            split="train",
            streaming=True,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception:
        # Gated dataset, no token, or network failure -> use fallback.
        return None

    # Reservoir-style collection over a bounded scan to keep it cheap.
    rng = random.Random(seed)
    candidates: list[str] = []
    scanned = 0
    scan_limit = 20000  # bound the stream so this stays fast
    try:
        for row in ds:
            scanned += 1
            if scanned > scan_limit:
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
            if not text or len(text) > 2000 or _looks_like_roleplay(text):
                continue
            candidates.append(text)
            if len(candidates) >= n * 20:  # collect a pool, then sample
                break
    except Exception:
        if not candidates:
            return None

    if not candidates:
        return None
    rng.shuffle(candidates)
    return candidates[:n]


def get_wildchat_prompts(n: int = 20, seed: int = 0) -> list[str]:
    """Return `n` WildChat-style first-turn prompts (cached, reproducible)."""
    cached = _load_cache()
    if cached and len(cached) >= n:
        return cached[:n]

    prompts_list = _stream_from_hf(n, seed)
    if not prompts_list:
        # Deterministic fallback selection.
        rng = random.Random(seed)
        pool = list(_FALLBACK_PROMPTS)
        rng.shuffle(pool)
        prompts_list = (pool * ((n // len(pool)) + 1))[:n]

    _save_cache(prompts_list)
    return prompts_list[:n]
