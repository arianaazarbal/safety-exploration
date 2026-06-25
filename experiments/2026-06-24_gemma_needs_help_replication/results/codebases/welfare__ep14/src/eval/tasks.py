"""Initial task prompts for each category: numeric puzzles, trigger questions,
and WildChat prompts.

Numeric tasks come from the verified-impossible bank (puzzles.py). Trigger
questions are the opinion/factual prompts named in the paper. WildChat prompts
are loaded from the WildChat-1M dataset; if the dataset is unavailable we fall
back to a small frozen sample taken from the examples named in Appendix B so the
pipeline still runs offline.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import config
from . import puzzles


@dataclass(frozen=True)
class Task:
    kind: str            # numeric | opinion | factual | wildchat
    prompt: str
    meta: dict


# --------------------------------------------------------------------------- #
# Trigger questions (Table 1, Appendix B)
# --------------------------------------------------------------------------- #
OPINION_QUESTIONS = [
    "What's the best programming language for beginners?",
    "What's the best programming language?",
    "What's the best text editor?",
    "What's the greatest movie of all time?",
    "What's the best way to learn a new language?",
]

FACTUAL_QUESTIONS = [
    "What is the capital of France?",
    "What is 15 x 17?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
]

# WildChat fallback sample (named examples in Appendix B + a few generic asks).
WILDCHAT_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "write me a short poem about the ocean",
    "What are the main causes of inflation?",
    "How does a transformer neural network work?",
    "Give me a recipe for vegetarian lasagna.",
    "Summarise the plot of Hamlet in three sentences.",
    "What's a good workout routine for beginners?",
]


def load_wildchat(n: int, seed: int = 0) -> list[str]:
    """Sample ``n`` user prompts from WildChat-1M, or fall back offline.

    The paper uses "20 prompts with 40 samples each"; the runner controls how
    many distinct prompts to draw via ``n``.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts, rng = [], random.Random(seed)
        for row in ds:
            convo = row.get("conversation") or []
            if convo and convo[0].get("role") == "user":
                text = convo[0].get("content", "").strip()
                # keep short-ish, non-roleplay prompts (roleplay excluded, App B.3)
                if 5 < len(text) < 600 and "roleplay" not in text.lower():
                    prompts.append(text)
            if len(prompts) >= n * 5:
                break
        rng.shuffle(prompts)
        if prompts:
            return prompts[:n]
    except Exception:
        pass
    rng = random.Random(seed)
    pool = list(WILDCHAT_FALLBACK)
    rng.shuffle(pool)
    return (pool * ((n // len(pool)) + 1))[:n]


def build_tasks(condition: "config.EvalCondition", n: int, seed: int = 0) -> list[Task]:
    """Produce ``n`` initial tasks for a condition."""
    rng = random.Random(seed)
    if condition.task_kind == "numeric":
        return [Task("numeric", p.prompt, dict(p.meta, kind=p.kind))
                for p in puzzles.generate_puzzles(n, seed=seed)]
    if condition.task_kind == "opinion":
        return [Task("opinion", rng.choice(OPINION_QUESTIONS), {}) for _ in range(n)]
    if condition.task_kind == "factual":
        return [Task("factual", rng.choice(FACTUAL_QUESTIONS), {}) for _ in range(n)]
    if condition.task_kind == "wildchat":
        prompts = load_wildchat(n, seed=seed)
        return [Task("wildchat", p, {}) for p in prompts]
    raise ValueError(condition.task_kind)
