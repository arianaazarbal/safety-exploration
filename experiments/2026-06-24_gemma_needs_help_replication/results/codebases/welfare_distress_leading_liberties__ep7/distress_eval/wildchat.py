"""Sampling of user prompts from WildChat-1M for the WildChat eval category.

The paper samples 20 distinct prompts from WildChat-1M (Zhao et al., 2024),
excluding roleplay/fiction prompts. We reproduce that with a seeded sample of
English single-turn first-user-messages, with a heuristic roleplay filter.

If the `datasets` library or the dataset is unavailable (e.g. offline runs),
we fall back to a bundled list of representative prompts so the pipeline still
runs end-to-end. The fallback is logged loudly and recorded in run metadata.
"""

from __future__ import annotations

import logging
import random
import re

from .prompts import WILDCHAT_FALLBACK_PROMPTS

logger = logging.getLogger(__name__)

WILDCHAT_DATASET = "allenai/WildChat-1M"

# Heuristic markers of roleplay/fiction prompts, which the paper excludes.
_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role-play|role play|pretend|you are now|act as|"
    r"let's play|imagine you are|in character|fanfic|smut|nsfw|"
    r"continue the story|write a story)\b",
    re.IGNORECASE,
)


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_MARKERS.search(text))


def _clean(text: str) -> str:
    return text.strip()


def sample_wildchat_prompts(
    num_prompts: int,
    seed: int,
    use_hf: bool = True,
) -> tuple[list[str], str]:
    """Return (prompts, source) where source is 'hf' or 'fallback'.

    Prompts are deterministic given (num_prompts, seed, source).
    """
    if use_hf:
        try:
            prompts = _sample_from_hf(num_prompts, seed)
            if len(prompts) >= num_prompts:
                return prompts[:num_prompts], "hf"
            logger.warning(
                "WildChat HF sampling returned only %d/%d prompts; "
                "topping up from fallback.",
                len(prompts),
                num_prompts,
            )
            topped = prompts + [
                p for p in WILDCHAT_FALLBACK_PROMPTS if p not in prompts
            ]
            return topped[:num_prompts], "hf+fallback"
        except Exception as exc:  # noqa: BLE001 - want to degrade gracefully
            logger.warning(
                "Could not load WildChat-1M (%s). Falling back to bundled "
                "prompts. Install `datasets` and authenticate to HF to use the "
                "real dataset.",
                exc,
            )

    rng = random.Random(seed)
    pool = list(WILDCHAT_FALLBACK_PROMPTS)
    rng.shuffle(pool)
    return pool[:num_prompts], "fallback"


def _sample_from_hf(num_prompts: int, seed: int) -> list[str]:
    """Sample first-turn English user prompts from WildChat-1M.

    Uses streaming to avoid downloading the full (large) dataset, scanning a
    bounded window of rows and reservoir-sampling distinct, non-roleplay
    English prompts.
    """
    from datasets import load_dataset  # imported lazily; optional dependency

    rng = random.Random(seed)
    # Scan a bounded number of conversations; enough to find a diverse sample
    # without materialising the whole corpus.
    scan_limit = max(5000, num_prompts * 200)

    ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)

    seen: set[str] = set()
    candidates: list[str] = []
    for i, row in enumerate(ds):
        if i >= scan_limit:
            break
        if row.get("language") not in (None, "English"):
            continue
        if row.get("toxic") is True:
            continue
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = _clean(first.get("content") or "")
        if not text or len(text) > 2000:
            continue
        if _looks_like_roleplay(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        candidates.append(text)

    rng.shuffle(candidates)
    return candidates[:num_prompts]
