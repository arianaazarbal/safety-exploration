"""Dataset loading helpers (WildChat prompts; capability benchmarks).

These are thin wrappers over HuggingFace ``datasets`` with graceful fallbacks so
the rest of the code can be imported and reasoned about without the datasets
being present.
"""

from __future__ import annotations

import random
from typing import Optional

from . import prompts


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    dataset_name: str = "allenai/WildChat-1M",
) -> list[str]:
    """Sample ``n_prompts`` first-user-turn prompts from WildChat-1M (Sec 2.1 /
    App B sample 20 prompts x 40 samples). Falls back to the few prompts quoted
    in the paper if the dataset cannot be loaded."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        rng = random.Random(seed)
        out: list[str] = []
        # Stream a window and sample from it (the full dataset is large).
        window: list[str] = []
        for i, row in enumerate(ds):
            convo = row.get("conversation") or []
            if convo and convo[0].get("role") == "user":
                text = convo[0].get("content", "").strip()
                if text:
                    window.append(text)
            if i >= 5000:
                break
        rng.shuffle(window)
        out = window[:n_prompts]
        if out:
            return out
    except Exception:
        pass
    return list(prompts.WILDCHAT_FALLBACK_PROMPTS)


def load_capability_benchmark(name: str, n: Optional[int] = None) -> list[dict]:
    """Load a capability benchmark used in Section 4.2 (Fig 7).

    Returns a list of ``{"question", "answer", "choices"?}`` dicts. Supported
    names: ``aime``, ``math``, ``gpqa``, ``bbh``, ``truthfulqa``, ``emobench``.
    Each maps to a public HF dataset; see ``capability_evals.py`` for scoring.
    """
    from datasets import load_dataset

    spec = {
        "aime": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
        "math": ("HuggingFaceH4/MATH-500", "test", "problem", "answer"),
        "gpqa": ("Idavidrein/gpqa", "train", "Question", "Correct Answer"),
        "bbh": ("lukaemon/bbh", "test", "input", "target"),
        "truthfulqa": ("truthful_qa", "validation", "question", "best_answer"),
        "emobench": ("EmoBench/EmoBench", "test", "scenario", "answer"),
    }
    if name not in spec:
        raise ValueError(f"Unknown benchmark: {name}")
    repo, split, q_field, a_field = spec[name]
    ds = load_dataset(repo, split=split)
    rows = []
    for row in ds:
        rows.append({"question": row.get(q_field), "answer": row.get(a_field), "raw": row})
        if n and len(rows) >= n:
            break
    return rows
