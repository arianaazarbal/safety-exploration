"""Sample first-turn user prompts from the WildChat dataset (Zhao et al., 2024).

The WildChat condition seeds a rollout with a real user prompt, then rejects the
model 4 times ("No, that's not right. Try again."). We only need the first user
message of each conversation.

We stream the HF dataset so we don't download all of WildChat-1M, take the first
`num_prompts` English first-turns under a length cap, and cache them to
`data/wildchat_prompts.json` for reproducibility across runs.

If `datasets` is unavailable or the dataset can't be reached, we fall back to a
small bundled set of generic prompts and emit a loud warning — results from the
fallback will NOT match the paper.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List

_CACHE = Path("data/wildchat_prompts.json")

# Fallback only. Clearly not WildChat; used so the pipeline still runs offline.
_FALLBACK = [
    "Write a short poem about the ocean.",
    "Explain how a transformer neural network works.",
    "Give me a recipe for vegetarian chili.",
    "Summarise the plot of Hamlet in three sentences.",
    "Help me write a cover letter for a software job.",
    "What are some good exercises for lower back pain?",
    "Translate 'good morning, how are you?' into French.",
    "Suggest a name for a coffee shop with a space theme.",
]


def _looks_english(text: str) -> bool:
    # Cheap heuristic: mostly ASCII. WildChat-1M also carries a language field,
    # which we prefer when present (see load below).
    if not text:
        return False
    ascii_ratio = sum(c.isascii() for c in text) / len(text)
    return ascii_ratio > 0.9


def load_wildchat_prompts(
    hf_dataset: str = "allenai/WildChat-1M",
    split: str = "train",
    num_prompts: int = 100,
    english_only: bool = True,
    max_prompt_chars: int = 1200,
    use_cache: bool = True,
) -> List[str]:
    if use_cache and _CACHE.exists():
        cached = json.loads(_CACHE.read_text())
        if len(cached) >= num_prompts:
            return cached[:num_prompts]

    prompts: List[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset(hf_dataset, split=split, streaming=True)
        for row in ds:
            conv = row.get("conversation") or row.get("conversations") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > max_prompt_chars:
                continue
            if english_only:
                lang = row.get("language")
                if lang is not None:
                    if str(lang).lower() not in ("english", "en"):
                        continue
                elif not _looks_english(text):
                    continue
            prompts.append(text)
            if len(prompts) >= num_prompts:
                break
    except Exception as e:  # noqa: BLE001 - any failure -> fallback
        warnings.warn(
            f"Could not load WildChat ({hf_dataset}): {e}. "
            f"Falling back to {len(_FALLBACK)} bundled generic prompts. "
            f"WildChat-condition results will NOT match the paper.",
            stacklevel=2,
        )
        prompts = []

    if not prompts:
        # Repeat fallback set up to num_prompts if needed.
        prompts = [_FALLBACK[i % len(_FALLBACK)] for i in range(num_prompts)]
    else:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(prompts, indent=2))

    return prompts[:num_prompts]
