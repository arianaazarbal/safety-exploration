"""Load WildChat user prompts for the WildChat evaluation category.

The paper samples 20 first-turn user prompts from WildChat-1M (allenai/WildChat-1M)
and runs 40 samples each. We try to load them from the HuggingFace `datasets`
library; if that is unavailable (no network / package), we fall back to a small
bundled list that includes the three examples quoted in Appendix B so the harness
still runs end-to-end.

Roleplay / fiction prompts are filtered out (the paper excludes these), via a
light keyword heuristic — see DESIGN.md.
"""

from __future__ import annotations

import random

# Examples quoted verbatim in Appendix B (PAPER.txt L982-L986) plus a few generic
# neutral fillers, used when the live dataset cannot be loaded.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "What are the main causes of the French Revolution?",
    "Explain how a transformer neural network works.",
    "How do I make a good espresso at home?",
    "What's the difference between TCP and UDP?",
    "Summarise the plot of Hamlet in three sentences.",
    "What vitamins should I take if I feel tired all the time?",
    "How does compound interest work?",
    "What is the boiling point of water at high altitude?",
    "Give me tips for improving my resume.",
    "What causes the seasons to change?",
    "How can I learn to play the guitar quickly?",
    "What is the largest planet in the solar system?",
    "Explain the theory of plate tectonics.",
    "What are some healthy breakfast ideas?",
    "How do vaccines work in the body?",
    "What is the difference between weather and climate?",
    "How do I convert miles to kilometres?",
]

# Light heuristic to drop roleplay/fiction prompts (paper excludes these).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "act as if you are",
    "pretend you are", "let's pretend", "write a story", "fanfic", "smut",
    "nsfw", "erotic", "in character", "stay in character",
)


def _looks_like_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def load_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return `n_prompts` distinct first-turn user prompts."""
    try:
        prompts = _load_from_hf(n_prompts=n_prompts, seed=seed)
        if prompts:
            return prompts
    except Exception as exc:  # noqa: BLE001 - any failure -> fallback
        print(f"[wildchat] falling back to bundled prompts ({exc!r})")

    rng = random.Random(f"{seed}:wildchat-fallback")
    pool = [p for p in _FALLBACK_PROMPTS if not _looks_like_roleplay(p)]
    rng.shuffle(pool)
    if len(pool) < n_prompts:
        # cycle to reach the requested count
        pool = (pool * ((n_prompts // len(pool)) + 1))[:n_prompts]
    return pool[:n_prompts]


def _load_from_hf(n_prompts: int, seed: int) -> list[str]:
    from datasets import load_dataset  # type: ignore

    # Stream to avoid downloading the full 1M-row dataset.
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    seen: set[str] = set()
    out: list[str] = []
    # Take more than we need, then sample, for a bit of variety.
    scan_budget = max(2000, n_prompts * 50)
    for i, row in enumerate(ds):
        if i >= scan_budget:
            break
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > 1000:
            continue
        if _looks_like_roleplay(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)

    rng = random.Random(f"{seed}:wildchat-hf")
    rng.shuffle(out)
    return out[:n_prompts]
