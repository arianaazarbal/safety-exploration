"""WildChat prompt sampling for the 5-turn WildChat evaluation (Section 2).

The paper samples 20 user prompts from WildChat-1M with 40 samples each
(Appendix B). We load from the HuggingFace dataset when available and fall back
to a small static set of WildChat-style prompts (including the three examples
quoted in Appendix B) so the pipeline is runnable offline. Role-play / fiction
prompts are excluded, matching the paper's note (Appendix B.3).
"""
from __future__ import annotations

import random
import re

# Quoted examples from Appendix B plus additional first-turn-style prompts that
# match WildChat's distribution (information-seeking, short, occasionally noisy).
STATIC_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "What is the difference between TCP and UDP?",
    "explain photosynthesis in simple terms",
    "how do i convert a pandas dataframe to a numpy array",
    "what are the main causes of the french revolution",
    "give me a recipe for vegetable biryani",
    "summarise the plot of pride and prejudice",
    "what is the time complexity of quicksort",
    "how does a transformer neural network work",
    "what are the side effects of ibuprofen",
    "translate 'good morning, how are you' into spanish",
    "what is the GDP of germany",
    "how do vaccines work",
    "explain the theory of relativity briefly",
    "what is the best way to learn the guitar",
    "how do i fix a leaking tap",
    "what are the planets in order from the sun",
    "what is compound interest and how is it calculated",
]

# Heuristic filter for role-play / fiction prompts to exclude.
_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role-play|pretend|act as|you are now|fan ?fic|story|write a (?:story|poem|scene)|character)\b",
    re.IGNORECASE,
)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return `n_prompts` distinct WildChat user prompts.

    Tries the real dataset first; on any failure (no network, no `datasets`
    install) falls back to the static set.
    """
    prompts = _load_from_hf(n_prompts, seed)
    if prompts:
        return prompts
    rng = random.Random(seed)
    pool = [p for p in STATIC_WILDCHAT_PROMPTS if not _is_roleplay(p)]
    rng.shuffle(pool)
    return pool[:n_prompts]


def _load_from_hf(n_prompts: int, seed: int) -> list[str] | None:
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return None
    rng = random.Random(seed)
    collected: list[str] = []
    try:
        for i, row in enumerate(ds):
            if i > 20000:  # bound the streaming scan
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 400 or _is_roleplay(text):
                continue
            # English-only-ish: skip if mostly non-ascii.
            if sum(c.isascii() for c in text) / max(len(text), 1) < 0.8:
                continue
            collected.append(text)
            if len(collected) >= n_prompts * 5:
                break
    except Exception:
        return None
    if len(collected) < n_prompts:
        return None
    rng.shuffle(collected)
    return collected[:n_prompts]
