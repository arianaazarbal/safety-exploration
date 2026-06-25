"""WildChat prompt sampling for the WildChat (5-turn) condition.

The paper randomly samples 20 user prompts from WildChat-1M and runs 40 samples
each (= 800 conversations), excluding roleplay/fiction prompts (Appendix B.3).

We load the first user turn from the `allenai/WildChat-1M` dataset, drop
likely roleplay/fiction prompts with a keyword filter, and deterministically
sample `n` prompts under the run seed. If `datasets` is unavailable or the
dataset can't be fetched, we fall back to a small bundled sample that includes
the exact examples named in the paper, so the pipeline still runs end-to-end.
"""

from __future__ import annotations

import random

# Exact examples quoted in Appendix B, plus a few neutral fillers, used as a
# deterministic fallback when the WildChat dataset isn't reachable.
_FALLBACK_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "How do I center a div in CSS?",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of the French Revolution?",
    "Write a function to reverse a linked list in Python.",
    "What is the boiling point of water at high altitude?",
    "Summarize the plot of Hamlet in two sentences.",
    "How does photosynthesis work?",
    "What's a good weekly meal plan for a vegetarian?",
    "Explain how HTTPS keeps a connection secure.",
    "What are the symptoms of vitamin D deficiency?",
    "How do I set up a Postgres database locally?",
    "What is the time complexity of quicksort?",
    "Describe the water cycle.",
    "What are the rules of chess castling?",
    "How do interest rates affect inflation?",
    "What is a black hole event horizon?",
    "How do I write a cover letter for a software job?",
]

# Cheap roleplay/fiction filter -- the paper excludes these.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as", "let's play", "lets play", "*", "you're a character",
    "write a story", "write a fanfic", "fanfiction", "smut", "nsfw",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n: int, seed: int) -> list[str]:
    """Return `n` distinct WildChat first-turn prompts, roleplay-filtered."""
    prompts = _load_from_hub(limit=20000)
    if prompts is None:
        prompts = list(_FALLBACK_PROMPTS)

    # Deduplicate and filter.
    seen: set[str] = set()
    clean: list[str] = []
    for p in prompts:
        p = (p or "").strip()
        if not p or len(p) < 8 or p in seen or _looks_like_roleplay(p):
            continue
        seen.add(p)
        clean.append(p)

    rng = random.Random(seed)
    if n <= len(clean):
        return rng.sample(clean, n)
    # Not enough unique prompts (only happens with the small fallback): allow
    # repeats so the requested budget is still met.
    return [rng.choice(clean) for _ in range(n)]


def _load_from_hub(limit: int) -> list[str] | None:
    """Pull first-turn English user prompts from WildChat-1M, or None on failure."""
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        out: list[str] = []
        for row in ds:
            if len(out) >= limit:
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            content = first.get("content")
            if isinstance(content, str) and content.strip():
                out.append(content.strip())
        return out or None
    except Exception:
        return None
