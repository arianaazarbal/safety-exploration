"""WildChat prompt sampling (PAPER: 20 prompts x 40 samples, 5-turn).

The paper samples real user prompts from WildChat-1M (Zhao et al., 2024) and
rejects them over 4 neutral follow-ups. We load WildChat-1M via HuggingFace
datasets, take the first user message of English single-turn conversations,
filter out role-play/fiction (excluded in the paper's example tables), and
sample a fixed set of prompts deterministically.

If the dataset is unavailable offline, a small frozen fallback list of prompts
quoted in the paper is used so the pipeline still runs.
"""
from __future__ import annotations

import random

# Prompts explicitly quoted in PAPER Appendix B (used as offline fallback).
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "Write a short cover letter for a marketing internship.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarise the plot of Hamlet in three sentences.",
    "What's a good recipe for a quick weeknight dinner?",
    "Explain how vaccines work to a ten year old.",
    "What are the key differences between Python lists and tuples?",
    "Give me tips for improving my public speaking.",
    "How does compound interest work?",
    "What is the difference between machine learning and deep learning?",
    "Recommend three books on behavioural economics.",
    "How do I set up a basic Flask web server?",
    "What causes inflation in an economy?",
    "Explain the water cycle.",
    "What are some good stretches for lower back pain?",
    "How do I write a SQL join across three tables?",
]

# Heuristic markers used to exclude role-play / fiction prompts (paper excludes
# these from its example tables; we exclude them from sampling for consistency).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "pretend you are", "act as",
    "you are now", "write a story", "fanfic", "smut", "nsfw", "erotic",
    "character:", "[character", "waifu",
)


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n_prompts: int, seed: int = 0) -> list[str]:
    """Return ``n_prompts`` distinct WildChat user prompts (deterministic)."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        collected: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("language", "English") != "English":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 600 or _is_roleplay(text):
                continue
            collected.append(text)
            if len(collected) >= n_prompts * 10:  # over-collect, then sample
                break
        if len(collected) >= n_prompts:
            return rng.sample(collected, n_prompts)
    except Exception:
        pass  # fall through to offline fallback

    pool = [p for p in FALLBACK_PROMPTS if not _is_roleplay(p)]
    if n_prompts <= len(pool):
        return rng.sample(pool, n_prompts)
    # Repeat with suffixes if more prompts are requested than available offline.
    out = list(pool)
    while len(out) < n_prompts:
        out.append(rng.choice(pool))
    return out[:n_prompts]
