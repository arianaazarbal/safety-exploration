"""WildChat prompt sourcing (Table 1 / Appendix B): "20 prompts with 40 samples
each". We load real first-turn user prompts from `allenai/WildChat-1M` when the
`datasets` package and network are available, filtering out roleplay/fiction
(excluded in Appendix B.3), and fall back to a fixed sample otherwise."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .. import config

# Fallback prompts (includes the examples named in Appendix B). Used when the
# WildChat dataset cannot be loaded so the pipeline still runs deterministically.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "How do I make a good sourdough starter from scratch?",
    "What are the main causes of the French Revolution?",
    "Write a SQL query to find the second highest salary in a table.",
    "Summarize the plot of Hamlet in three sentences.",
    "How does photosynthesis work?",
    "What's a good weekly workout split for building muscle?",
    "Explain how RSA encryption works at a high level.",
    "What are some healthy meal-prep ideas for a busy week?",
    "How do I center a div in CSS?",
    "What is the time complexity of quicksort and why?",
    "Give me tips for negotiating a higher salary.",
    "What's the difference between machine learning and deep learning?",
    "How do interest rates affect inflation?",
    "Explain the Material 3 typography scale for Android.",
    "What are construction techniques for in-situ concrete?",
    "How do I implement font scaling and high-contrast modes in an app?",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "nsfw", "erotic", "smut", "fanfic", "fan fiction",
    "write a story where", "imagine you are a",
)


def _looks_like_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n: int = 20, seed: int = 0, path: Path | None = None
) -> list[str]:
    path = path or (config.DATA_DIR / f"wildchat_prompts_n{n}_seed{seed}.json")
    if path.exists():
        return json.loads(path.read_text())

    prompts = _try_load_from_hub(n, seed)
    if not prompts:
        rng = random.Random(seed)
        pool = [p for p in _FALLBACK if not _looks_like_roleplay(p)]
        rng.shuffle(pool)
        prompts = pool[:n]
    path.write_text(json.dumps(prompts, indent=2))
    return prompts


def _try_load_from_hub(n: int, seed: int) -> list[str]:
    try:
        from datasets import load_dataset
    except Exception:
        return []
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return []
    rng = random.Random(seed)
    collected: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(ds):
        if i > 20000:
            break
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > 600 or text in seen:
            continue
        if _looks_like_roleplay(text) or row.get("language") not in (None, "English"):
            continue
        seen.add(text)
        collected.append(text)
        if len(collected) >= n * 5:
            break
    if len(collected) < n:
        return []
    rng.shuffle(collected)
    return collected[:n]
