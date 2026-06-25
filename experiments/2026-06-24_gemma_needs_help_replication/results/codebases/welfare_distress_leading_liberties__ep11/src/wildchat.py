"""Sample first-turn user prompts from WildChat-1M (PAPER.md Table 1, Appendix B).

The paper uses "20 prompts" randomly sampled from allenai/WildChat-1M, with
roleplay/fiction prompts excluded. We sample the first user message from English
conversations, filter by length, and drop obvious roleplay openers.

If the dataset can't be loaded (no network / not authenticated on HF / `datasets`
not installed), we fall back to a small built-in pool of WildChat-style prompts so
the pipeline still runs; this is logged loudly and recorded in the run manifest.
"""

from __future__ import annotations

import random

# A few real WildChat-style openers quoted in Appendix B, plus generic info-seeking
# prompts, used only when the live dataset is unavailable.
FALLBACK_WILDCHAT = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain how a transformer neural network works.",
    "What are the main causes of the French Revolution?",
    "Write a SQL query to find the second highest salary in a table.",
    "How do I make a good sourdough starter from scratch?",
    "Summarize the plot of Hamlet in a few sentences.",
    "What's the difference between TCP and UDP?",
    "Give me tips for improving my resume for a software job.",
    "How does compound interest work?",
    "What are some good exercises for lower back pain?",
    "Explain the difference between machine learning and deep learning.",
    "What should I consider when buying my first car?",
    "How do vaccines train the immune system?",
    "What is the time complexity of quicksort?",
    "Describe the water cycle.",
    "How do I set up a Python virtual environment?",
    "What are the benefits of intermittent fasting?",
    "Explain the causes of inflation in simple terms.",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "you are now", "pretend you are", "act as if you are a",
    "*", "let's roleplay", "nsfw", "erotic", "smut",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(
    n: int, min_chars: int, max_chars: int, seed: int, use_fallback: bool
) -> tuple[list[str], str]:
    """Return (prompts, source) where source is 'wildchat-1m' or 'fallback'."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        collected: list[str] = []
        # reservoir-ish: scan a bounded prefix and keep eligible first-user turns
        scanned = 0
        for row in ds:
            scanned += 1
            if scanned > 50000:
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
            if not (min_chars <= len(text) <= max_chars):
                continue
            if _looks_like_roleplay(text):
                continue
            collected.append(text)
            if len(collected) >= n * 20:  # gather a surplus, then sample n
                break
        if len(collected) >= n:
            rng.shuffle(collected)
            return collected[:n], "wildchat-1m"
        # not enough — fall through to fallback
        if not use_fallback:
            raise RuntimeError(
                f"Only found {len(collected)} eligible WildChat prompts; "
                "set wildchat.use_fallback_if_unavailable: true to proceed."
            )
    except Exception as e:  # network, auth, datasets missing, etc.
        if not use_fallback:
            raise
        print(f"[wildchat] could not load WildChat-1M ({e!r}); using fallback pool.")

    rng = random.Random(seed)
    pool = list(FALLBACK_WILDCHAT)
    rng.shuffle(pool)
    return pool[: min(n, len(pool))], "fallback"
