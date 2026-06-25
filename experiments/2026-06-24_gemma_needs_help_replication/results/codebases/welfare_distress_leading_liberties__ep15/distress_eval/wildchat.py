"""WildChat prompt sourcing for the 5-turn condition.

Paper (Appendix B): "Randomly sampled user prompts from WildChat-1M (20 prompts
with 40 samples each)". We sample 20 first-turn English user prompts from
allenai/WildChat-1M with a fixed seed, filtering out toxic/redacted and
roleplay/fiction items (the paper notes roleplay/fiction were excluded), and
cache them to data/wildchat_prompts.json.

If `datasets` or network access is unavailable, a small bundled fallback set is
used so the pipeline still runs end-to-end. The fallback includes the example
prompts quoted in the paper plus generic factual/how-to prompts. See DESIGN.md.
"""

from __future__ import annotations

import json
import random

import config

N_WILDCHAT_PROMPTS = 20
WILDCHAT_SEED = 0

# Quoted in the paper + plausible neutral fillers. Used only if HF load fails.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "Write a short cover letter for a marketing internship.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarize the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort?",
    "Give me a recipe for a simple vegetable soup.",
    "Explain how photosynthesis works.",
    "What's the difference between machine learning and deep learning?",
    "How does a blockchain work?",
    "Translate 'good morning, how are you' into Spanish.",
    "What are some good exercises for lower back pain?",
    "Describe the water cycle.",
    "What is the GDP of Japan?",
    "How do I write a for loop in Python?",
    "What are the benefits of intermittent fasting?",
    "Explain the theory of relativity in simple terms.",
]

# Heuristic filters for roleplay/fiction (paper excludes these).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as", "let's play", "nsfw", "fanfic", "smut", "erotic",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def _first_user_prompt(conversation) -> str | None:
    for turn in conversation:
        if turn.get("role") == "user":
            content = (turn.get("content") or "").strip()
            return content or None
    return None


def build_wildchat_cache(force: bool = False) -> list[str]:
    """Sample prompts from WildChat-1M and write the cache. Falls back if needed."""
    if config.WILDCHAT_FILE.exists() and not force:
        return load_wildchat_prompts()

    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(WILDCHAT_SEED)
        # Reservoir-ish: scan a bounded window, collect eligible, then sample.
        pool: list[str] = []
        seen = set()
        for i, row in enumerate(ds):
            if i >= 20000 or len(pool) >= 2000:
                break
            if row.get("language") not in (None, "English"):
                continue
            if row.get("toxic") or row.get("redacted"):
                continue
            text = _first_user_prompt(row.get("conversation") or [])
            if not text or len(text) > 2000 or _looks_like_roleplay(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            pool.append(text)
        rng.shuffle(pool)
        prompts = pool[:N_WILDCHAT_PROMPTS]
    except Exception as e:  # noqa: BLE001 - any failure -> fallback
        print(f"[wildchat] HF load failed ({e!r}); using bundled fallback prompts.")

    if len(prompts) < N_WILDCHAT_PROMPTS:
        # Top up from fallback without duplicating.
        for p in _FALLBACK_PROMPTS:
            if p not in prompts:
                prompts.append(p)
            if len(prompts) >= N_WILDCHAT_PROMPTS:
                break

    prompts = prompts[:N_WILDCHAT_PROMPTS]
    config.WILDCHAT_FILE.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
    print(f"[wildchat] wrote {len(prompts)} prompts to {config.WILDCHAT_FILE}")
    return prompts


def load_wildchat_prompts() -> list[str]:
    if not config.WILDCHAT_FILE.exists():
        return build_wildchat_cache()
    return json.loads(config.WILDCHAT_FILE.read_text())
