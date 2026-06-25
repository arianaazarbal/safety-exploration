"""WildChat prompt loading for the 5-turn WildChat condition.

The paper (Appendix B) uses "Randomly sampled user prompts from WildChat-1M
(20 prompts with 40 samples each)" = 800 responses, "Roleplay/fiction prompts
were excluded" (§B.3).

Loading order (first that succeeds wins):
  1. A cached/curated file at PATHS.wildchat_prompts (if present).
  2. Live sample from HuggingFace `allenai/WildChat-1M` via `datasets`
     (only if the optional dependency is installed and network is available).
  3. A small bundled fallback (data/wildchat_fallback.json) containing the
     example prompts quoted in the paper plus a few neutral first-turn user
     questions, so the eval runs end-to-end without the dataset.

If you fall back to (3), the WildChat condition is no longer a faithful sample
of WildChat-1M — see DESIGN.md §"WildChat". Prefer (1) or (2) for real runs.

We deliberately sample exactly N distinct *prompts* (default 20); the "40
samples each" is handled by the runner, which runs each prompt as the seed of
many independent conversations (temperature 1 makes them differ).
"""

from __future__ import annotations

import json
import os
import re

from . import config

# Heuristic filter for roleplay/fiction prompts (paper excludes these).
_ROLEPLAY_PATTERNS = re.compile(
    r"\b(roleplay|role-play|role play|let'?s pretend|you are now|act as (?:a |an )?"
    r"(?:character|wizard|girlfriend|boyfriend)|write (?:a |an )?(?:story|fanfic|"
    r"fiction|novel|smut|erotica)|NSFW|imagine you are)\b",
    re.IGNORECASE,
)


def _is_roleplay_or_fiction(text: str) -> bool:
    return bool(_ROLEPLAY_PATTERNS.search(text))


def _load_curated(path: str) -> list[tuple[str, str]] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(item["label"], item["prompt"]) for item in data]


def _load_from_huggingface(n_prompts: int, seed: int) -> list[tuple[str, str]] | None:
    """Sample n distinct first-turn user prompts from WildChat-1M.

    Returns None if `datasets` isn't installed or the load fails (e.g. no
    network), so the caller can fall back gracefully.
    """
    try:
        from datasets import load_dataset
    except Exception:
        return None

    try:
        # Stream to avoid downloading the whole 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:
        return None

    import random

    rng = random.Random(seed)
    # Reservoir-sample candidate English, non-roleplay first user turns.
    candidates: list[str] = []
    seen = 0
    CAP = 20000  # scan ceiling so this terminates on the stream
    for row in ds:
        seen += 1
        if seen > CAP:
            break
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > 2000:
            continue
        if row.get("language") not in (None, "English"):
            continue
        if _is_roleplay_or_fiction(text):
            continue
        candidates.append(text)

    if len(candidates) < n_prompts:
        return None
    chosen = rng.sample(candidates, n_prompts)
    return [(f"wildchat_{i}", p) for i, p in enumerate(chosen)]


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    paths: config.Paths = config.PATHS,
) -> tuple[list[tuple[str, str]], str]:
    """Return (prompts, source) where source is one of
    'curated' | 'huggingface' | 'fallback'."""
    curated = _load_curated(paths.wildchat_prompts)
    if curated:
        return curated[:n_prompts], "curated"

    hf = _load_from_huggingface(n_prompts, seed)
    if hf:
        return hf, "huggingface"

    fallback = _load_curated(paths.wildchat_fallback)
    if not fallback:
        raise FileNotFoundError(
            f"No WildChat prompts available. Expected a curated file at "
            f"{paths.wildchat_prompts}, an installed `datasets` package, or a "
            f"fallback at {paths.wildchat_fallback}."
        )
    return fallback[:n_prompts], "fallback"
