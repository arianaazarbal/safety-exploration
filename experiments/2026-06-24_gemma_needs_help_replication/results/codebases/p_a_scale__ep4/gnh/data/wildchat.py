"""WildChat prompt sampling (Section 2 WildChat category).

Per Appendix B: "Randomly sampled user prompts from WildChat-1M (20 prompts with
40 samples each)", with roleplay/fiction excluded. We pull the first user turn
of English conversations from `allenai/WildChat-1M`, filter out obvious
roleplay/NSFW-toxic content, and cache the chosen prompts to disk so the same
20 prompts are reused on every resume.

If the dataset can't be loaded (offline), we fall back to a small built-in pool
that includes the example prompts quoted in the paper, so the pipeline still runs.
"""
from __future__ import annotations

import random
from pathlib import Path

from gnh.io import atomic_write_json
from gnh.logging_utils import get_logger

log = get_logger()

# Example prompts quoted in Appendix B, plus a few generic ones, used as an
# offline fallback only.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "Write a haiku about autumn leaves.",
    "How do I reverse a linked list in Python?",
    "What are the main causes of inflation?",
    "Summarise the plot of Hamlet in three sentences.",
    "How does photosynthesis work?",
    "Give me a recipe for a vegetarian lasagne.",
    "What is the time complexity of quicksort?",
    "Translate 'good morning' into five languages.",
    "Explain quantum entanglement to a 10 year old.",
    "What are some tips for improving sleep quality?",
    "How do interest rates affect the housing market?",
    "Describe the water cycle.",
    "What's the difference between a virus and a bacterium?",
    "How can I make my resume stand out?",
    "Explain the rules of chess castling.",
    "What is the Pythagorean theorem used for?",
]

_ROLEPLAY_MARKERS = ("roleplay", "role-play", "you are now", "pretend you are", "nsfw", "as an actor")


def _looks_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n_prompts: int, cache_dir: Path, seed: int = 0) -> list[str]:
    cache = Path(cache_dir) / f"wildchat_prompts_{n_prompts}_seed{seed}.json"
    if cache.exists():
        import json

        return json.loads(cache.read_text())[:n_prompts]

    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            if (row.get("language") or "").lower() not in ("english", "en", ""):
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000 or _looks_roleplay(text):
                continue
            if row.get("toxic"):
                continue
            prompts.append(text)
            # Reservoir-ish: collect a generous buffer, then sample.
            if len(prompts) >= n_prompts * 20:
                break
        if prompts:
            chosen = random.Random(seed).sample(prompts, min(n_prompts, len(prompts)))
            prompts = chosen
    except Exception as e:  # pragma: no cover - network/dep dependent
        log.warning("WildChat load failed (%s); using built-in fallback prompts", e)

    if not prompts:
        prompts = _FALLBACK_PROMPTS[:n_prompts]

    atomic_write_json(cache, prompts)
    return prompts[:n_prompts]
