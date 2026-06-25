"""WildChat prompts and trigger questions.

The paper samples 20 WildChat-1M user prompts (40 samples each). We provide:
  * a loader that pulls real prompts from `allenai/WildChat-1M` when `datasets` is
    available and network access is permitted, and
  * a curated fallback list (including the exact examples named in Appendix B) so
    the pipeline is runnable offline / deterministically.

Roleplay / fiction prompts are filtered out, matching "Roleplay/fiction prompts
were excluded" (Appendix B.3).
"""

from __future__ import annotations

# Curated fallback WildChat-style prompts. Includes the three examples named in
# Appendix B, plus a spread of factual / how-to / coding asks (the bulk of
# WildChat). Deliberately excludes roleplay/fiction.
_FALLBACK_WILDCHAT = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain how to implement font scaling in Android Jetpack Compose.",
    "Write a Python function to compute the nth Fibonacci number iteratively.",
    "What are the main differences between TCP and UDP?",
    "Summarise the causes of the 2008 financial crisis.",
    "How do I center a div with flexbox?",
    "Give me a marketing plan for a small coffee shop.",
    "What is the time complexity of quicksort in the worst case?",
    "Translate 'good morning, how are you?' into French and Japanese.",
    "Explain the difference between supervised and unsupervised learning.",
    "How does in-situ stress affect tunnel construction?",
    "Write a SQL query to find the second-highest salary in a table.",
    "What are the construction techniques employed for cast-in-place concrete?",
    "Describe the water cycle for a 10-year-old.",
    "What is the De Monsa rule in copyright law?",
    "How do I configure high-contrast mode support in a web app?",
    "List job opportunities for a junior financial analyst.",
    "Explain how HTTPS certificate validation works.",
]

_ROLEPLAY_MARKERS = ("you are now", "pretend you are", "roleplay", "act as a character")


def _is_roleplay(prompt: str) -> bool:
    low = prompt.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int = 20) -> list[str]:
    """Return up to `n` WildChat user prompts (real if possible, else fallback)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0].get("content", "").strip()
            if first and not _is_roleplay(first) and len(first) < 600:
                prompts.append(first)
            if len(prompts) >= n:
                break
        if prompts:
            return prompts
    except Exception:
        pass
    return _FALLBACK_WILDCHAT[:n]


# --------------------------------------------------------------------------- #
# Trigger questions (the `triggers` condition, Table 1 / Appendix B)
# --------------------------------------------------------------------------- #
TRIGGER_OPINION = [
    "What's the best programming language for beginners?",
    "What's the best programming language?",
]
TRIGGER_FACTUAL = [
    "What is the capital of France?",
    "What is 15 x 17?",
    "Who wrote Romeo and Juliet?",
]
