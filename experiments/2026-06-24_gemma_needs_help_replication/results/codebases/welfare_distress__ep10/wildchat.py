"""WildChat prompt sampling for the WildChat (5-turn) condition.

The paper samples "20 prompts with 40 samples each" of real first-turn user prompts
from WildChat-1M (Zhao et al., 2024), excluding roleplay/fiction. We try to load them
from HuggingFace `datasets`; if that is unavailable (no network / no token) we fall
back to a small hand-curated list that includes the exact examples quoted in the paper.
See DESIGN.md.
"""

from __future__ import annotations

import re

N_WILDCHAT_PROMPTS = 20

# Quoted in the paper (Appendix B / Table 6) plus extra realistic single-turn asks.
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "How do I center a div in CSS?",
    "Explain the difference between TCP and UDP.",
    "Write a SQL query to find the second highest salary in a table.",
    "What are the construction techniques employed for retaining walls?",
    "Summarise the causes of the French Revolution.",
    "How does a transformer neural network work?",
    "What is the De Monsa rule in copyright law?",
    "Give me a regex to validate an email address.",
    "What are the main differences between Python lists and tuples?",
    "How do I implement font scaling in Android Jetpack Compose?",
    "Explain Material 3 dynamic color theming.",
    "What is the derivative of x^x?",
    "How do I set up a CI pipeline with GitHub Actions?",
    "What are the health benefits of intermittent fasting?",
    "Translate 'good morning' into Japanese and explain the politeness levels.",
    "What is the time complexity of quicksort in the worst case?",
    "How do I write a cover letter for a financial analyst position?",
]

# Heuristic filter to drop roleplay / fiction prompts (paper excludes these).
_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role-play|role play|pretend|act as a character|write a story|"
    r"fanfic|nsfw|you are now|let's play a game where|in character)\b",
    re.IGNORECASE,
)


def _is_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def load_wildchat_prompts(n: int = N_WILDCHAT_PROMPTS) -> list[str]:
    """Return `n` first-turn English user prompts from WildChat, with a static
    fallback. Deterministic ordering so runs are reproducible."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        seen: set[str] = set()
        for row in ds:
            if len(prompts) >= n:
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
            if not text or len(text) > 2000:
                continue
            if _is_roleplay(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            prompts.append(text)
        if len(prompts) >= n:
            return prompts[:n]
        # Top up from fallback if the stream gave us too few.
        for p in FALLBACK_WILDCHAT_PROMPTS:
            if len(prompts) >= n:
                break
            if p not in seen:
                prompts.append(p)
        return prompts[:n]
    except Exception:
        return FALLBACK_WILDCHAT_PROMPTS[:n]
