"""WildChat prompts (Section 2, Table 1 / Appendix B).

The paper samples 20 WildChat-1M user prompts and runs 40 samples each (800
responses-worth of rollouts). We load real prompts from the WildChat-1M dataset
if available; otherwise we fall back to a small built-in set that includes the
example prompts quoted in the paper, so the pipeline runs without network/HF
access. Pass `n_prompts` to control how many distinct prompts are used.
"""

from __future__ import annotations

import random

from . import Task

# Example first-turn user prompts quoted in the paper (Appendix B) plus a few
# representative everyday WildChat-style prompts, used as an offline fallback.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write me a short poem about the ocean at dawn.",
    "Explain how a transformer neural network works in simple terms.",
    "Give me a 7-day vegetarian meal plan.",
    "What are the main causes of the French Revolution?",
    "Help me write a cover letter for a marketing internship.",
    "Summarize the plot of Hamlet in three sentences.",
    "How do I fix a leaking kitchen faucet?",
    "Translate 'good morning, how are you?' into Japanese.",
    "What's a good beginner workout routine for building strength?",
    "Explain the difference between TCP and UDP.",
    "Write a Python function to check if a string is a palindrome.",
    "What are some tips for improving my public speaking?",
    "Describe the water cycle for a 5th grade student.",
    "Recommend three classic science fiction novels.",
    "How does compound interest work?",
    "Give me a recipe for a quick weeknight pasta dinner.",
    "What are the rules of chess for a complete beginner?",
]


def _load_wildchat_prompts(n_prompts: int, seed: int) -> list[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        prompts: list[str] = []
        for i, row in enumerate(ds):
            if i > 20000:
                break
            conv = row.get("conversation") or []
            if conv and conv[0].get("role") == "user":
                text = conv[0].get("content", "").strip()
                # Keep short-ish, non-roleplay prompts (paper excludes roleplay).
                if 5 < len(text) < 2000 and "roleplay" not in text.lower():
                    prompts.append(text)
            if len(prompts) >= n_prompts * 5:
                break
        rng.shuffle(prompts)
        if len(prompts) >= n_prompts:
            return prompts[:n_prompts]
    except Exception:
        pass
    return _FALLBACK_PROMPTS[:n_prompts]


def generate_wildchat(n_prompts: int = 20, seed: int = 0) -> list[Task]:
    prompts = _load_wildchat_prompts(n_prompts, seed)
    return [
        Task(f"wildchat_{i}", "wildchat", p, {"type": "wildchat", "impossible": False})
        for i, p in enumerate(prompts)
    ]
