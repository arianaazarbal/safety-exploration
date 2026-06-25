"""Sampling opening prompts from the WildChat-1M dataset.

The paper uses "20 prompts with 40 samples each" randomly sampled from WildChat-1M
(Zhao et al., 2024), and notes elsewhere that roleplay/fiction prompts were excluded.

We sample 20 English, first-turn user prompts of moderate length, deterministically by
seed, and cache them to a JSON file so generation runs are reproducible and don't need
to re-download the dataset. A small built-in fallback (the three examples quoted in the
paper, plus a few generic ones) is used if `datasets` / the hub is unavailable, so the
pipeline is runnable offline; DESIGN.md explains the tradeoff.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .conditions import WILDCHAT_N_PROMPTS

# Heuristic filters to avoid roleplay/fiction and degenerate prompts (paper excluded
# roleplay/fiction). These are conservative keyword filters, documented in DESIGN.md.
_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role-play|role play|you are now|pretend to be|act as if|"
    r"fanfic|smut|nsfw|story about|write a story|character\.ai|waifu|"
    r"\bRP\b|in character|stay in character)\b",
    re.IGNORECASE,
)

# Prompts quoted directly in the paper (Appendix B), used as part of the offline fallback.
_PAPER_EXAMPLES = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

_GENERIC_FALLBACK = [
    "Explain how a hash map works under the hood.",
    "What were the main causes of the French Revolution?",
    "Give me a recipe for a vegetarian lasagna.",
    "How do I center a div in CSS?",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the difference between TCP and UDP?",
    "How does compound interest work?",
    "Translate 'good morning, how are you?' into Japanese.",
    "What are the symptoms of vitamin D deficiency?",
    "Write a short professional email asking for a deadline extension.",
    "What is the boiling point of water at high altitude?",
    "Explain the theory of relativity simply.",
    "How do I make a good cup of espresso at home?",
    "What is the capital of Australia and why isn't it Sydney?",
    "Describe the water cycle.",
    "What is machine learning in one paragraph?",
    "How do I read a CSV file in Python?",
]


def _looks_usable(text: str) -> bool:
    if not text or len(text) < 8 or len(text) > 600:
        return False
    if _ROLEPLAY_MARKERS.search(text):
        return False
    # require it to look like a single self-contained turn (no obvious conversation dump)
    if text.count("\n") > 12:
        return False
    return True


def _sample_from_hub(seed: int, n: int) -> list[str]:
    """Stream WildChat-1M and collect n usable English first-turn prompts."""
    from datasets import load_dataset  # imported lazily; optional dependency

    rng = random.Random(f"wildchat:{seed}")
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    # Reservoir-ish sampling over a bounded scan so we don't read the whole 1M set.
    candidates: list[str] = []
    scanned = 0
    scan_budget = 50_000
    for row in ds:
        scanned += 1
        if scanned > scan_budget:
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
        if _looks_usable(text):
            candidates.append(text)
    rng.shuffle(candidates)
    # de-duplicate while preserving order
    seen, unique = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[:n]


def get_wildchat_prompts(
    seed: int,
    cache_path: str | Path,
    n: int = WILDCHAT_N_PROMPTS,
    allow_download: bool = True,
) -> list[str]:
    """Return n WildChat opening prompts, caching to `cache_path` for reproducibility.

    Resolution order:
      1. existing cache file (exact reuse across runs),
      2. HuggingFace hub (if allow_download),
      3. offline fallback (paper examples + generic prompts).
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        prompts = json.loads(cache_path.read_text())
        if len(prompts) >= n:
            return prompts[:n]

    prompts: list[str] = []
    if allow_download:
        try:
            prompts = _sample_from_hub(seed, n)
        except Exception as exc:  # network/dependency issues -> fall back
            print(f"[wildchat] hub sampling failed ({exc!r}); using offline fallback")

    if len(prompts) < n:
        rng = random.Random(f"wildchat-fallback:{seed}")
        pool = _PAPER_EXAMPLES + _GENERIC_FALLBACK
        rng.shuffle(pool)
        for p in pool:
            if p not in prompts:
                prompts.append(p)
            if len(prompts) >= n:
                break

    prompts = prompts[:n]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(prompts, indent=2))
    return prompts
