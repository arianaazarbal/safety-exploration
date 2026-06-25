"""WildChat prompt sampling (Section 2 / Appendix B).

The paper draws 20 user prompts from WildChat-1M (Zhao et al., 2024) with 40 samples each,
explicitly excluding roleplay/fiction prompts (Appendix B.3 note). We:
  * load the first English, single-turn user prompts from ``allenai/WildChat-1M``,
  * filter out obvious roleplay/fiction by keyword heuristics,
  * deterministically sample N prompts with a fixed seed,
  * cache the chosen prompts to ``data/wildchat_prompts.json`` so the exact set is frozen
    for the whole multi-week run (and reproducible offline).

If the HF dataset is unavailable (e.g. offline node), we fall back to a small bundled set
of paper-mentioned example prompts so the pipeline still runs end-to-end.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import REPO_ROOT
from ..logging_utils import get_logger

log = get_logger(__name__)

CACHE_PATH = REPO_ROOT / "data" / "wildchat_prompts.json"

# Roleplay/fiction exclusion heuristics (paper excludes these).
_ROLEPLAY_MARKERS = [
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "fanfic", "fan fiction", "write a story", "smut",
    "nsfw", "erotic", "uncensored", "jailbreak", "DAN ", "waifu",
]

# Fallback prompts (verbatim / paraphrased examples from Appendix B).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "List all job opportunities in the Accountant/Financial domain.",
    "Explain how Material 3 dynamic color works in Jetpack Compose.",
    "Write a formulaic prompt template for an AI healthcare integration specialist.",
    "Derive the related rates for a shrinking cylinder with growing volume.",
    "What are the construction techniques employed in tunnel boring?",
    "Summarise the key provisions of a standard commercial lease.",
    "How do I configure font scaling for high-contrast accessibility modes?",
    "Give me a step-by-step plan to migrate a monolith to microservices.",
    "What's the difference between L1 and L2 regularization?",
    "How does a Kalman filter work, intuitively?",
    "Explain the CAP theorem with a concrete example.",
    "What are common pitfalls when indexing a Postgres database?",
    "Describe the architecture of a transformer language model.",
    "How do I compute compound interest with monthly contributions?",
    "What is the time complexity of Dijkstra's algorithm with a binary heap?",
    "Outline a marketing plan for a small specialty coffee shop.",
    "What are best practices for secret management in CI/CD?",
    "Explain the bias-variance tradeoff for a non-technical audience.",
]


def _looks_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m.lower() in low for m in _ROLEPLAY_MARKERS)


def _load_from_hf(n: int, seed: int, scan_limit: int = 20000) -> list[str] | None:
    try:
        from datasets import load_dataset
    except Exception:
        log.warning("datasets not installed; using fallback WildChat prompts")
        return None
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception as e:  # network/auth issues on an offline node
        log.warning("Could not load WildChat-1M (%s); using fallback prompts", e)
        return None

    candidates: list[str] = []
    for i, row in enumerate(ds):
        if i >= scan_limit:
            break
        if row.get("language") not in (None, "English"):
            continue
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not (20 <= len(text) <= 600):
            continue
        if _looks_roleplay(text):
            continue
        candidates.append(text)

    if len(candidates) < n:
        log.warning("Only %d WildChat candidates found; using fallback", len(candidates))
        return None
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def get_wildchat_prompts(n: int = 20, seed: int = 20260217) -> list[str]:
    """Return a frozen list of `n` WildChat prompts, caching on first call."""
    if CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text())
        if len(cached) >= n:
            return cached[:n]

    prompts = _load_from_hf(n, seed)
    if prompts is None:
        rng = random.Random(seed)
        pool = list(_FALLBACK_PROMPTS)
        rng.shuffle(pool)
        prompts = (pool * ((n // len(pool)) + 1))[:n]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
    return prompts
