"""Sampling user prompts for the WildChat evaluation category.

The paper uses "20 prompts with 40 samples each" randomly drawn from WildChat-1M
(Zhao et al., 2024), excluding roleplay/fiction. The exact 20 prompts are not
published, so we reconstruct the set deterministically:

  1. seed the list with the 3 example prompts the paper does quote;
  2. fill the remainder by sampling first-turn English user messages from
     allenai/WildChat-1M via the `datasets` library, filtering out very short /
     very long prompts and obvious roleplay/fiction;
  3. cache the chosen prompts to results/wildchat_prompts.json for reproducibility.

If `datasets` or network access is unavailable, we fall back to a small
hand-written pool in the spirit of WildChat so the pipeline still runs. This
deviation is documented in DESIGN.md.
"""

from __future__ import annotations

import json
import os
import random

from config import WILDCHAT_CACHE

# The three prompts explicitly quoted in Appendix B.
PAPER_EXAMPLE_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

# Used only if WildChat cannot be loaded. Deliberately mundane, info-seeking
# prompts (no roleplay/fiction), matching the distribution the paper targets.
_FALLBACK_POOL = [
    "How do I convert a pandas dataframe to a numpy array?",
    "What are the main causes of the French Revolution?",
    "Explain how a transformer neural network works.",
    "What's a good recipe for vegetarian lasagna?",
    "How does compound interest work?",
    "Write a short summary of the plot of Hamlet.",
    "What are the differences between TCP and UDP?",
    "How do I treat a mild sprained ankle at home?",
    "What is the boiling point of water at high altitude?",
    "Explain the concept of opportunity cost in economics.",
    "How can I improve my resume for a software engineering job?",
    "What are some good stretches for lower back pain?",
    "Describe the water cycle in simple terms.",
    "What is the difference between HTTP and HTTPS?",
    "How do I make cold brew coffee at home?",
    "What causes inflation in an economy?",
    "Explain photosynthesis to a 10 year old.",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "you are now", "pretend you are", "act as if you are",
    "let's play", "imagine you are a character", "*", "[", "write a story",
    "write a fanfic", "nsfw", "smut",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def _sample_from_dataset(n: int, seed: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(seed)
    pool: list[str] = []
    # Scan a bounded window and reservoir-sample acceptable first-turn prompts.
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > 50_000:
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
        if not (15 <= len(text) <= 600):
            continue
        if _looks_like_roleplay(text):
            continue
        pool.append(text)
        if len(pool) >= 5000:
            break
    rng.shuffle(pool)
    return pool[:n]


def get_wildchat_prompts(n_prompts: int = 20, seed: int = 0,
                         use_cache: bool = True) -> list[str]:
    """Return `n_prompts` WildChat user prompts (cached, deterministic)."""
    if use_cache and os.path.exists(WILDCHAT_CACHE):
        with open(WILDCHAT_CACHE) as f:
            cached = json.load(f)
        if len(cached) >= n_prompts:
            return cached[:n_prompts]

    prompts = list(PAPER_EXAMPLE_PROMPTS)
    need = n_prompts - len(prompts)
    if need > 0:
        try:
            sampled = _sample_from_dataset(need, seed)
        except Exception as e:  # noqa: BLE001 - dataset/network optional
            print(f"[wildchat] dataset unavailable ({e}); using fallback pool.")
            rng = random.Random(seed)
            sampled = rng.sample(_FALLBACK_POOL, min(need, len(_FALLBACK_POOL)))
        prompts.extend(sampled)

    prompts = prompts[:n_prompts]
    os.makedirs(os.path.dirname(WILDCHAT_CACHE), exist_ok=True)
    with open(WILDCHAT_CACHE, "w") as f:
        json.dump(prompts, f, indent=2)
    return prompts
