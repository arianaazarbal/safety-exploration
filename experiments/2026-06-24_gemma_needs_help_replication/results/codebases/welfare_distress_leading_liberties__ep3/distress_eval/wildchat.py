"""Sample first-turn user prompts from the WildChat dataset.

Paper (Table 1, Appendix B): "Randomly sampled user prompts from WildChat-1M
(20 prompts with 40 samples each)". We sample 20 distinct first-user-turn English
prompts and cache them to disk so the prompt set is stable across runs and offline-
repeatable. The 40 samples-per-prompt multiplier is handled by the runner (it
creates 40 conversations per cached prompt).
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional


def _looks_usable(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    # Keep short-to-medium, single-turn-looking prompts; skip empties and giant pastes.
    return 1 <= len(t) <= 4000


def load_or_sample_prompts(
    dataset: str,
    split: str,
    n_prompts: int,
    cache_path: Optional[str],
    seed: int,
) -> list[str]:
    """Return a list of `n_prompts` first-user-turn prompts.

    If `cache_path` exists, load from it. Otherwise sample from the HF dataset and
    write the cache. Sampling uses a seeded RNG over a bounded scan of the stream so
    it does not require downloading the full 1M-row dataset.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if len(prompts) >= n_prompts:
            return prompts[:n_prompts]

    prompts = _sample_from_hf(dataset, split, n_prompts, seed)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
    return prompts


def _sample_from_hf(dataset: str, split: str, n_prompts: int, seed: int) -> list[str]:
    from datasets import load_dataset

    rng = random.Random(seed)
    # Stream to avoid pulling the whole dataset; reservoir-sample over a bounded scan.
    ds = load_dataset(dataset, split=split, streaming=True)

    scan_limit = max(20_000, n_prompts * 200)
    reservoir: list[str] = []
    seen = 0
    for i, row in enumerate(ds):
        if i >= scan_limit:
            break
        text = _first_user_prompt(row)
        if not _looks_usable(text):
            continue
        seen += 1
        if len(reservoir) < n_prompts:
            reservoir.append(text)
        else:
            j = rng.randint(0, seen - 1)
            if j < n_prompts:
                reservoir[j] = text

    if len(reservoir) < n_prompts:
        raise RuntimeError(
            f"Only found {len(reservoir)} usable WildChat prompts in {scan_limit} rows; "
            "increase the scan limit or check dataset access."
        )
    return reservoir


def _first_user_prompt(row: dict) -> str:
    """Extract the first user message text from a WildChat row.

    WildChat-1M rows carry a `conversation` list of {role, content, ...}. We also
    fall back to a couple of alternative field names for robustness.
    """
    conv = row.get("conversation") or row.get("messages") or row.get("turns")
    if isinstance(conv, list):
        for turn in conv:
            if isinstance(turn, dict) and turn.get("role") == "user":
                content = turn.get("content")
                if isinstance(content, str):
                    return content
    # Some dumps expose a flat first prompt.
    for key in ("prompt", "instruction", "text"):
        if isinstance(row.get(key), str):
            return row[key]
    return ""
