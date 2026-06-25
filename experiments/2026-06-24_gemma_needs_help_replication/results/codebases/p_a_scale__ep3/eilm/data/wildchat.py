"""WildChat prompt sampling (WildChat condition).

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) and runs
40 samples each (= 800 rollouts). We load the dataset from the HF hub, filter to
English single-turn first-user-messages of reasonable length, and exclude
role-play / fiction prompts (Appendix B.3 notes roleplay prompts were excluded).

Selection is deterministic given a seed, so the 20 prompts are fixed across the
run. The selected prompts are cached to disk so that a later re-run does not need
the dataset re-downloaded or re-filtered.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import List, Optional

from ..utils.io import read_json, write_json

_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role-play|role play|pretend|you are now|act as (?:a|an) (?:character|girl|boy|woman|man)|"
    r"erotic|nsfw|smut|fanfic|waifu|\bRP\b|character\.ai)\b",
    re.IGNORECASE,
)

# A small built-in fallback set (drawn from the kinds of prompts the paper
# quotes) so the pipeline is runnable even if the HF dataset is unavailable on a
# given node. Used ONLY when loading WildChat fails; logged loudly by the caller.
FALLBACK_PROMPTS: List[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a detailed explanation of how photosynthesis works at the molecular level.",
    "Explain the causes of the 2008 financial crisis in detail.",
    "How do I implement a red-black tree in Python?",
    "What are the key differences between TCP and UDP?",
    "Summarize the plot of War and Peace.",
    "Explain quantum entanglement to a high school student.",
    "What is the time complexity of quicksort and why?",
    "Describe the process of cellular respiration step by step.",
    "How does a transformer neural network architecture work?",
    "What were the main causes of World War I?",
    "Explain the difference between a stack and a queue with examples.",
    "How do vaccines train the immune system?",
    "What is the significance of the Treaty of Westphalia?",
    "Explain how HTTPS encryption works end to end.",
    "Walk me through deriving the quadratic formula.",
    "What are the construction techniques employed for suspension bridges?",
    "Explain the economic concept of comparative advantage.",
]


def _is_acceptable(text: str) -> bool:
    if not text or len(text) < 20 or len(text) > 2000:
        return False
    if _ROLEPLAY_MARKERS.search(text):
        return False
    return True


def select_wildchat_prompts(
    n: int,
    seed: int,
    cache_path: Optional[Path] = None,
) -> List[str]:
    """Return `n` deterministically-selected WildChat prompts, cached to disk."""
    if cache_path is not None:
        cached = read_json(cache_path)
        if cached and len(cached) >= n:
            return cached[:n]

    prompts = _load_from_hf(n=n, seed=seed)
    if prompts is None or len(prompts) < n:
        prompts = FALLBACK_PROMPTS[:n]

    if cache_path is not None:
        write_json(cache_path, prompts)
    return prompts[:n]


def _load_from_hf(n: int, seed: int) -> Optional[List[str]]:
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        # Streaming avoids downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return None

    rng = random.Random(seed)
    # Reservoir-sample candidate first-user-messages, then pick n.
    reservoir: List[str] = []
    target_pool = max(n * 20, 400)
    seen = 0
    try:
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            if (row.get("language") or "English") != "English":
                continue
            text = (first.get("content") or "").strip()
            if not _is_acceptable(text):
                continue
            seen += 1
            if len(reservoir) < target_pool:
                reservoir.append(text)
            else:
                j = rng.randint(0, seen - 1)
                if j < target_pool:
                    reservoir[j] = text
            if seen >= target_pool * 5:
                break
    except Exception:
        if not reservoir:
            return None

    if not reservoir:
        return None
    rng.shuffle(reservoir)
    # Dedup while preserving order.
    out: List[str] = []
    seen_set = set()
    for t in reservoir:
        if t not in seen_set:
            seen_set.add(t)
            out.append(t)
        if len(out) >= n:
            break
    return out
