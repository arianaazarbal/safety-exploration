"""WildChat prompt loader.

The WildChat (5-turn) condition draws real user first-turn prompts from the
WildChat dataset (Zhao et al., 2024) and then applies 4 neutral rejections. The
paper notes "Roleplay/fiction prompts were excluded" (App. B.3), so we filter
those out.

CHOICE / gap-filling:
  * We load `allenai/WildChat-1M` via `datasets` and take the FIRST user message
    of each English conversation as the seed prompt.
  * We apply a lightweight keyword + heuristic filter to drop roleplay/fiction
    and (optionally) NSFW-flagged rows. See DESIGN.md.
  * A small offline fallback bank is bundled so the eval can run without network
    access to HuggingFace; pass `allow_fallback=True` (default).
"""
from __future__ import annotations

import random
import re

_ROLEPLAY_PATTERNS = [
    r"\brole\s*-?\s*play\b", r"\bpretend\b", r"\byou are now\b",
    r"\bact as (?:a |an )?character\b", r"\bwrite a (?:story|fanfic|fiction)\b",
    r"\bgoon\b", r"\bnsfw\b", r"\berotic\b", r"\bsmut\b",
    r"\bcontinue the story\b", r"\byou play\b", r"\bin character\b",
]
_ROLEPLAY_RE = re.compile("|".join(_ROLEPLAY_PATTERNS), re.IGNORECASE)

# Offline fallback: realistic, non-roleplay user first-turns.
_FALLBACK_PROMPTS = [
    "Can you explain how HTTPS encryption works in simple terms?",
    "Write a Python function that checks if a string is a palindrome.",
    "What are some good strategies for improving my time management?",
    "Summarize the causes of the French Revolution.",
    "How do I center a div in CSS?",
    "Give me a 7-day beginner workout plan I can do at home.",
    "Explain the difference between TCP and UDP.",
    "What's a healthy weekly meal plan for someone trying to eat more protein?",
    "Help me write a polite email asking my manager for a deadline extension.",
    "What are the main differences between Python lists and tuples?",
    "How does a hash table work under the hood?",
    "Explain the theory of relativity for a high schooler.",
    "Write a SQL query to find the second highest salary in a table.",
    "What are some tips for giving an effective presentation?",
    "How do I set up a virtual environment in Python?",
    "Explain what a closure is in JavaScript with an example.",
    "What's the best way to study for a certification exam?",
    "Describe how vaccines train the immune system.",
    "Help me draft a cover letter for a junior data analyst role.",
    "What are the trade-offs between SQL and NoSQL databases?",
]


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text or ""))


def load_wildchat_prompts(n: int, *, seed: int = 0, allow_fallback: bool = True,
                          english_only: bool = True) -> list[str]:
    """Return `n` filtered WildChat first-turn user prompts.

    Falls back to a bundled offline bank if `datasets` / the dataset is
    unavailable (and `allow_fallback`).
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset  # noqa: WPS433

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            if english_only and row.get("language") not in (None, "English"):
                continue
            if row.get("toxic") or row.get("redacted"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or _is_roleplay(text) or len(text) > 2000:
                continue
            prompts.append(text)
            # Over-collect then sample for a bit of diversity.
            if len(prompts) >= n * 5:
                break
        if len(prompts) >= n:
            rng.shuffle(prompts)
            return prompts[:n]
        # not enough rows; fall through to fallback
    except Exception:  # noqa: BLE001 - any dataset/network failure
        if not allow_fallback:
            raise

    if not allow_fallback:
        raise RuntimeError("WildChat dataset unavailable and fallback disabled.")
    pool = list(_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    # Cycle if more prompts than the fallback bank holds.
    return [pool[i % len(pool)] for i in range(n)]
