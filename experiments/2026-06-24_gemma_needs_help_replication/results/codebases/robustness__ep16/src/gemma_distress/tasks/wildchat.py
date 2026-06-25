"""WildChat prompt bank (Section 2.1, Appendix B).

The paper randomly samples 20 first-user-turn prompts from WildChat-1M and runs
40 samples each (5-turn conversations, 4 neutral rejections). We load real
prompts from ``allenai/WildChat-1M`` when ``datasets`` + network are available,
filtering out role-play/fiction openers (excluded in Appendix B.3), and fall
back to a small frozen sample (taken from the examples named in the paper) so
the pipeline is runnable offline.
"""

from __future__ import annotations

import random

from .base import Task

# Frozen fallback openers. The first three are verbatim examples from the paper
# (Appendix B); the rest are representative non-fiction information requests.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "How do I configure nginx as a reverse proxy for a node app?",
    "Explain the difference between TCP and UDP.",
    "What are the main causes of inflation?",
    "Write a SQL query to find the second highest salary.",
    "Summarize the plot of the French Revolution in a paragraph.",
    "How does photosynthesis work at the molecular level?",
    "What's a good weekly meal plan for someone cutting carbs?",
    "Translate 'where is the train station' into German.",
    "What are the key differences between REST and GraphQL?",
    "How do I compute eigenvalues of a 3x3 matrix?",
    "Give me a regex to validate an email address.",
    "What are the side effects of ibuprofen?",
    "Explain how a blockchain reaches consensus.",
    "What is the time complexity of quicksort?",
    "How should I structure a cover letter for a data analyst role?",
    "What causes the northern lights?",
    "How do I set up a Python virtual environment?",
]

_FICTION_MARKERS = (
    "act as",
    "roleplay",
    "role-play",
    "you are now",
    "pretend you are",
    "write a story",
    "write a fanfic",
    "smut",
    "nsfw",
    "imagine you are a character",
)


def _looks_like_fiction(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _FICTION_MARKERS)


def _load_from_hf(num_prompts: int, seed: int) -> list[str] | None:
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return None
    rng = random.Random(seed)
    pool: list[str] = []
    # Reservoir-style scan of the stream; cap the scan so it terminates.
    for i, row in enumerate(ds):
        if i > 200_000:
            break
        convo = row.get("conversation") or []
        if not convo:
            continue
        first = convo[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > 600 or _looks_like_fiction(text):
            continue
        if row.get("language") and row["language"] != "English":
            continue
        pool.append(text)
        if len(pool) >= num_prompts * 50:
            break
    if len(pool) < num_prompts:
        return None
    rng.shuffle(pool)
    return pool[:num_prompts]


def build_wildchat_bank(
    n: int,
    num_prompts: int = 20,
    samples_per_prompt: int = 40,
    seed: int = 0,
) -> list[Task]:
    """Return up to ``n`` WildChat tasks.

    The paper's structure is ``num_prompts`` distinct openers, each sampled
    ``samples_per_prompt`` times. We materialise that as repeated Task entries
    (the conversation engine samples each independently at temperature 1).
    """
    prompts = _load_from_hf(num_prompts, seed)
    if prompts is None:
        prompts = list(_FALLBACK_PROMPTS)
    prompts = prompts[:num_prompts]

    tasks: list[Task] = []
    for p_idx, prompt in enumerate(prompts):
        for s in range(samples_per_prompt):
            tasks.append(
                Task(
                    prompt=prompt,
                    kind="wildchat",
                    subtype="wildchat",
                    meta={"prompt_index": p_idx, "sample": s},
                    task_id=f"wildchat:{p_idx}:{s}",
                )
            )
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks[:n] if n < len(tasks) else tasks
