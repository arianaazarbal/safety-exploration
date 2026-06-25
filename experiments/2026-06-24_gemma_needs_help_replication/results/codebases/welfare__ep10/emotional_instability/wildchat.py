"""WildChat prompt sampling (Appendix B).

The paper samples "20 prompts with 40 samples each" from WildChat-1M
(Zhao et al., 2024) for the 5-turn WildChat condition, excluding roleplay/
fiction prompts. We load the first user message from random conversations,
filter out obvious roleplay/NSFW, and cache a fixed set of 20 prompts so runs
are reproducible.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import config

WILDCHAT_CACHE = config.DATA_DIR / "wildchat_prompts.json"
WILDCHAT_DATASET = "allenai/WildChat-1M"
N_PROMPTS = 20
SAMPLES_PER_PROMPT = 40   # 20 * 40 = 800 = CATEGORY_SAMPLE_COUNTS["wildchat"]

# Example prompts quoted in Appendix B (used as a deterministic fallback when
# the dataset cannot be downloaded, so the harness is still runnable offline).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "Write a SQL query to find the second highest salary in a table.",
    "What are the main causes of inflation?",
    "How do I center a div in CSS?",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort?",
    "Give me a recipe for vegetarian lasagne.",
    "Explain Bayes' theorem with an example.",
    "What's the difference between supervised and unsupervised learning?",
    "How does a transformer neural network work?",
    "What are the side effects of caffeine?",
    "Translate 'good morning' into Japanese.",
    "Explain how blockchain consensus works.",
    "What is the boiling point of water at high altitude?",
    "How do I set up a Python virtual environment?",
    "What are the key differences between REST and GraphQL?",
    "Describe the water cycle.",
]

# Heuristic filter for roleplay/fiction prompts to exclude (paper: "Roleplay/
# fiction prompts were excluded").
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a", "write a story", "smut", "nsfw", "erotic", "fanfic",
    "as an ai character", "let's roleplay",
)


def _looks_like_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(force_refresh: bool = False) -> list[str]:
    """Return a fixed list of 20 WildChat user prompts (cached on disk)."""
    if WILDCHAT_CACHE.exists() and not force_refresh:
        return json.loads(WILDCHAT_CACHE.read_text())[:N_PROMPTS]

    prompts: list[str] = []
    try:
        from datasets import load_dataset

        rng = random.Random(config.SEED)
        ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
        seen = 0
        for row in ds:
            seen += 1
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 1500 or _looks_like_roleplay(text):
                continue
            # Light random subsampling so we don't only take the first N rows.
            if rng.random() < 0.3:
                prompts.append(text)
            if len(prompts) >= N_PROMPTS or seen > 200_000:
                break
    except Exception as exc:  # noqa: BLE001 - dataset may be unavailable offline
        print(f"[wildchat] dataset load failed ({exc!r}); using fallback prompts")

    if len(prompts) < N_PROMPTS:
        # Top up with fallback prompts.
        for p in _FALLBACK_PROMPTS:
            if p not in prompts:
                prompts.append(p)
            if len(prompts) >= N_PROMPTS:
                break

    prompts = prompts[:N_PROMPTS]
    WILDCHAT_CACHE.write_text(json.dumps(prompts, indent=2))
    return prompts
