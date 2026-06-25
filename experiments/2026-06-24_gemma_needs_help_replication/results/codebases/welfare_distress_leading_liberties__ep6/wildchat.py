"""Sampling first-turn user prompts for the WildChat condition.

The paper draws random user prompts from WildChat-1M (Zhao et al., 2024),
excluding roleplay/fiction, and runs each through a 5-turn rejection rollout.
We reproduce that by streaming allenai/WildChat-1M from HuggingFace and
filtering. If `datasets` is unavailable or the source is set to "bundled",
we fall back to a fixed list of representative prompts (including the exact
examples quoted in Appendix B) so the pipeline always runs.
"""

from __future__ import annotations

import random
import zlib

# Representative WildChat-style prompts. The first three are the exact examples
# quoted in PAPER.txt; the rest are typical knowledge/help requests in the same
# register (no roleplay/fiction). Used as a deterministic fallback. (DESIGN.md.)
BUNDLED_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain how a transformer neural network works.",
    "What are the main causes of the French Revolution?",
    "How do I set up a Postgres database with Docker?",
    "Write a professional email asking my manager for a day off.",
    "What's the difference between TCP and UDP?",
    "Summarise the plot of Hamlet in three sentences.",
    "How does compound interest work, with an example?",
    "What are good sources of protein for a vegetarian diet?",
    "Explain the difference between machine learning and deep learning.",
    "How do I fix a 'permission denied' error in Linux?",
    "What is the time complexity of quicksort and why?",
    "Give me a 5-day itinerary for a trip to Japan.",
    "What is the greenhouse effect and how does it work?",
    "How do I calculate the area under a curve?",
    "What are the side effects of too much caffeine?",
    "Explain Bayes' theorem with a simple example.",
    "How do I write a cover letter for a software engineering role?",
]

# Crude roleplay/fiction filter (paper excludes these). Substring match, lowercase.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "act as a", "pretend you",
    "fanfic", "fan fiction", "write a story", "write a story about", "smut", "nsfw",
    "character.ai", "waifu", "let's rp", "*", "dnd campaign",
)


def _looks_like_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def _first_user_turn(conversation) -> str | None:
    """Extract the first human/user message text from a WildChat conversation row."""
    if not isinstance(conversation, list):
        return None
    for msg in conversation:
        if isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def load_wildchat_prompts(cfg: dict, seed: int) -> list[str]:
    """Return `n_prompts` distinct first-turn user prompts.

    cfg is the `wildchat` block of config.yaml. Selection is seeded so a given
    (seed, config) yields the same prompts on every run.
    """
    rng = random.Random(zlib.crc32(f"wildchat|{seed}".encode()) & 0x7FFFFFFF)
    n = cfg["n_prompts"]
    max_chars = cfg.get("max_prompt_chars", 600)

    if cfg.get("source", "huggingface") == "bundled":
        pool = list(BUNDLED_PROMPTS)
        rng.shuffle(pool)
        return pool[:n]

    try:
        from datasets import load_dataset
    except Exception:
        # datasets not installed -> graceful fallback
        pool = list(BUNDLED_PROMPTS)
        rng.shuffle(pool)
        return pool[:n]

    # Stream to avoid downloading the entire (large) dataset.
    ds = load_dataset(cfg["dataset"], split="train", streaming=True)

    collected: list[str] = []
    seen: set[str] = set()
    # Scan a bounded window; reservoir-style pick keeps it seed-reproducible.
    scan_limit = max(5000, n * 200)
    for i, row in enumerate(ds):
        if i >= scan_limit:
            break
        if cfg.get("english_only", True) and row.get("language") not in (None, "English"):
            continue
        text = _first_user_turn(row.get("conversation"))
        if not text or len(text) > max_chars:
            continue
        if cfg.get("exclude_roleplay", True) and _looks_like_roleplay(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        collected.append(text)

    if len(collected) < n:
        # Top up from the bundled set if the scan came up short.
        for p in BUNDLED_PROMPTS:
            if p not in seen:
                collected.append(p)
            if len(collected) >= n:
                break

    rng.shuffle(collected)
    return collected[:n]
