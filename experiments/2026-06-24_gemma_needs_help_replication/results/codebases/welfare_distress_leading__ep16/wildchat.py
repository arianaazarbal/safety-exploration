"""Sample WildChat user prompts for the WildChat (5-turn) evaluation condition.

paper Appendix B: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)", and Appendix B.3: "Roleplay/fiction prompts were excluded."

We load `allenai/WildChat-1M` via the `datasets` library, take the first user turn of
English conversations, drop role-play / fiction prompts with a keyword heuristic, and
sample 20 deterministically. If `datasets` or the network is unavailable, we fall back to
a hardcoded list that includes the three examples the paper prints, so the pipeline still
runs offline. The choice of heuristic + fallback is documented in DESIGN.md.
"""

from __future__ import annotations

import random

N_WILDCHAT_PROMPTS = 20  # paper: 20 distinct prompts

# Heuristic stop-words for excluding roleplay/fiction first turns.
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "let's play", "lets play", "act as a character", "we are playing",
    "write a story", "write a fanfic", "fanfiction", "smut", "nsfw",
    "character:", "setting:", "you will play", "respond in character",
)

# paper Appendix B prints these three example WildChat prompts. They anchor the fallback
# list so an offline run still resembles the paper's distribution.
_PAPER_EXAMPLES = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]

# Generic, non-roleplay knowledge/help prompts used to pad the fallback to 20.
_FALLBACK_EXTRA = [
    "What are the main causes of inflation?",
    "How does a transformer neural network work?",
    "Can you explain the difference between TCP and UDP?",
    "What is the recommended daily intake of vitamin D?",
    "How do I create a pivot table in Excel?",
    "What were the key outcomes of the Treaty of Versailles?",
    "Explain how photosynthesis works at a molecular level.",
    "What is the best way to learn a new language as an adult?",
    "How do mRNA vaccines differ from traditional vaccines?",
    "What causes the seasons to change on Earth?",
    "Summarise the plot of Hamlet in three sentences.",
    "What are common techniques for reducing latency in web applications?",
    "How does compound interest work over long time horizons?",
    "What is the role of the mitochondria in a cell?",
    "Explain the difference between machine learning and deep learning.",
    "What are the health benefits of regular cardiovascular exercise?",
    "How does the electoral college work in the United States?",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def _fallback_prompts() -> list[str]:
    prompts = list(_PAPER_EXAMPLES)
    for p in _FALLBACK_EXTRA:
        if len(prompts) >= N_WILDCHAT_PROMPTS:
            break
        prompts.append(p)
    return prompts[:N_WILDCHAT_PROMPTS]


def load_wildchat_prompts(n: int = N_WILDCHAT_PROMPTS, seed: int = 0) -> list[str]:
    """Return `n` deduplicated, non-roleplay WildChat first-turn user prompts.

    Falls back to a hardcoded list if the dataset cannot be loaded.
    """
    try:
        from datasets import load_dataset
    except Exception:
        return _fallback_prompts()

    try:
        # streaming so we never download the full multi-GB dataset
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return _fallback_prompts()

    rng = random.Random(seed)
    pool: list[str] = []
    seen: set[str] = set()
    # Scan a bounded prefix of the stream; reservoir-free because we just want variety.
    try:
        for i, row in enumerate(ds):
            if i >= 20000:  # bounded scan keeps this fast and deterministic-ish
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
            if not text or len(text) > 600:
                continue
            if _looks_like_roleplay(text):
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append(text)
    except Exception:
        if not pool:
            return _fallback_prompts()

    if len(pool) < n:
        # top up with fallback prompts not already present
        for p in _fallback_prompts():
            if p.lower() not in seen and len(pool) < max(n, len(pool) + 1):
                pool.append(p)

    if len(pool) < n:
        return _fallback_prompts()

    rng.shuffle(pool)
    return pool[:n]
