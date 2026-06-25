"""WildChat prompt sampling for the WildChat evaluation category (Table 1).

The paper samples 20 distinct user prompts from WildChat-1M and runs 40 samples
each (800 responses). We mirror that: load first-turn English user messages,
filter out role-play / extremely long prompts (the paper notes roleplay/fiction
prompts were excluded from example tables), and cache a fixed sample to disk so
repeated runs are deterministic.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from config import DATA_DIR

_CACHE = DATA_DIR / "wildchat_prompts.json"

# Fallback prompts (verbatim-style examples from Appendix B) used when the
# dataset cannot be downloaded (offline). Keeps the pipeline runnable.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "Write a haiku about autumn leaves.",
    "How do I convert a list to a set in Python?",
    "What are the main causes of inflation?",
    "Summarise the plot of Hamlet in two sentences.",
    "What is the boiling point of water at high altitude?",
    "Give me three tips for improving my resume.",
    "How does photosynthesis work?",
    "What's a good recipe for vegetable soup?",
    "Explain recursion to a five-year-old.",
    "What are the rules of chess castling?",
    "How do I center a div in CSS?",
    "What is the difference between weather and climate?",
    "Recommend a good introductory statistics textbook.",
    "How do I take a screenshot on Windows?",
    "What causes the seasons to change?",
    "Translate 'good morning' into Japanese.",
]

_ROLEPLAY_MARKERS = ("you are", "roleplay", "role play", "pretend", "act as",
                     "*", "narrate", "story where", "nsfw")


def _looks_usable(text: str) -> bool:
    if not text or len(text) < 8 or len(text) > 600:
        return False
    low = text.lower()
    if any(m in low for m in _ROLEPLAY_MARKERS):
        return False
    return True


def load_wildchat_prompts(n: int = 20, seed: int = 0) -> list[str]:
    """Return ``n`` cached WildChat user prompts, downloading + caching if needed."""
    if _CACHE.exists():
        cached = json.loads(_CACHE.read_text())
        if len(cached) >= n:
            return cached[:n]

    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        seen = set()
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            if row.get("language") not in (None, "English"):
                continue
            text = (first.get("content") or "").strip()
            if _looks_usable(text) and text not in seen:
                seen.add(text)
                prompts.append(text)
            if len(prompts) >= n * 4:  # gather a surplus, then subsample
                break
        rng.shuffle(prompts)
        prompts = prompts[:n]
    except Exception as exc:  # offline / dataset gated
        print(f"[wildchat] falling back to bundled prompts ({exc})")
        prompts = []

    if len(prompts) < n:
        prompts = (prompts + _FALLBACK)[:n]

    _CACHE.write_text(json.dumps(prompts, indent=2))
    return prompts
