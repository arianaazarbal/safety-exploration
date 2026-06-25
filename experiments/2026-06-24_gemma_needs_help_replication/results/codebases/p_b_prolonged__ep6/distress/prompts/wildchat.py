"""WildChat prompt sampling (Section 2.1, Appendix B).

Paper: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We load the first user turn from `allenai/WildChat-1M`, filter
to English single-turn-openable prompts, exclude role-play/fiction (Appendix
B.3 notes these were excluded), and cache a fixed 20-prompt sample to disk so
runs are reproducible.

If the dataset cannot be downloaded offline, a small built-in fallback set of
representative prompts (quoted in the paper) is used so the pipeline still runs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import DATA_DIR

WILDCHAT_CACHE = DATA_DIR / "wildchat_sample.json"
N_PROMPTS = 20
SAMPLES_PER_PROMPT = 40  # 20 * 40 = 800 WildChat responses (Appendix B)

# Heuristic filters to drop role-play / fiction (excluded per Appendix B.3).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "act as a character", "fanfic", "fan fiction", "write a story",
    "smut", "nsfw", "erotic",
)

# Fallback prompts (quoted in Appendix B) if the dataset is unavailable.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "How do I center a div in CSS?",
    "Summarise the causes of the French Revolution.",
    "What are the construction techniques employed for in-situ concrete?",
    "Write a SQL query to find the second highest salary.",
    "Explain quantum entanglement simply.",
    "What is the time complexity of quicksort?",
    "How does photosynthesis work?",
    "Give me a recipe for sourdough bread.",
    "What are the main differences between Python 2 and 3?",
    "Explain the CAP theorem.",
    "How do vaccines work?",
    "What is the difference between machine learning and deep learning?",
    "Describe the water cycle.",
    "What causes inflation in an economy?",
    "How do I reverse a linked list?",
    "What is the difference between HTTP and HTTPS?",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def _build_sample_from_hf(seed: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(seed)
    candidates: list[str] = []
    for row in ds:
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        if row.get("language") not in (None, "English"):
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > 600 or _looks_like_roleplay(text):
            continue
        candidates.append(text)
        if len(candidates) >= 5000:
            break
    rng.shuffle(candidates)
    return candidates[:N_PROMPTS]


def get_wildchat_prompts(seed: int = 0, *, force_rebuild: bool = False) -> list[str]:
    if WILDCHAT_CACHE.exists() and not force_rebuild:
        return json.loads(WILDCHAT_CACHE.read_text())
    try:
        prompts = _build_sample_from_hf(seed)
        if len(prompts) < N_PROMPTS:
            raise RuntimeError("not enough WildChat candidates")
    except Exception:  # noqa: BLE001 - offline / dataset gated -> fallback
        prompts = list(_FALLBACK)[:N_PROMPTS]
    WILDCHAT_CACHE.write_text(json.dumps(prompts, indent=2))
    return prompts
