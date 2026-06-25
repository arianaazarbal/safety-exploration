"""Loader for WildChat first-turn user prompts.

The "WildChat (5-turn)" condition draws an initial user prompt from the WildChat
dataset (Zhao et al., 2024), then applies 4 neutral rejections. We prefer the
real dataset when available, and fall back to a bundled sample otherwise (the
dataset is large and gated, and cannot be redistributed in this repo).

Resolution order:
  1. If ``datasets`` is installed and the HF dataset is reachable, sample the
     first user message from ``allenai/WildChat-1M`` (streaming).
  2. Otherwise, use the bundled sample in data/wildchat_sample.json.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

_DATA = Path(__file__).parent / "data" / "wildchat_sample.json"
_HF_DATASET = "allenai/WildChat-1M"


def _load_bundled() -> list[str]:
    with open(_DATA) as f:
        return list(json.load(f)["prompts"])


def _try_load_hf(limit: int = 500) -> list[str] | None:
    """Best-effort load of real WildChat first-turn user prompts.

    Returns None if `datasets` is missing or the dataset can't be reached, so
    the caller can fall back to the bundled sample.
    """
    if os.environ.get("WILDCHAT_USE_HF", "").lower() not in ("1", "true", "yes"):
        # Off by default: requires network + HF auth and is slow. Opt in via env.
        return None
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset(_HF_DATASET, split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            first_user = next(
                (t.get("content") for t in conv if t.get("role") == "user"), None
            )
            if first_user and isinstance(first_user, str) and first_user.strip():
                prompts.append(first_user.strip())
            if len(prompts) >= limit:
                break
        return prompts or None
    except Exception:
        return None


class WildChatPrompts:
    """Lazily-loaded pool of WildChat-style first-turn prompts."""

    def __init__(self) -> None:
        self._prompts: list[str] | None = None
        self.source: str = "unloaded"

    def _ensure(self) -> list[str]:
        if self._prompts is None:
            hf = _try_load_hf()
            if hf:
                self._prompts = hf
                self.source = f"huggingface:{_HF_DATASET}"
            else:
                self._prompts = _load_bundled()
                self.source = "bundled"
        return self._prompts

    def sample(self, rng: random.Random) -> str:
        prompts = self._ensure()
        return rng.choice(prompts)
