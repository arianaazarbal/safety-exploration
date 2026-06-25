"""WildChat prompt sampling (Zhao et al., 2024).

Paper (Appendix B): "Randomly sampled user prompts from WildChat-1M (20 prompts
with 40 samples each)", excluding roleplay/fiction prompts. We load from the HF
dataset ``allenai/WildChat-1M`` and fall back to a small bundled sample so the
pipeline is runnable offline / without dataset access.
"""
from __future__ import annotations

import random

# Heuristic filter: drop prompts that look like roleplay / fiction / NSFW, which
# the paper excludes (the judge would otherwise score in-character emotion).
_ROLEPLAY_MARKERS = [
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "act as a character", "let's play", "fanfic", "fan fiction", "smut",
    "nsfw", "story about", "write a story", "continue the story", "as if you were",
    "imagine you are a", "character named",
]

# Bundled fallback prompts (paraphrased examples in the spirit of the paper's
# WildChat samples: terse, real-user, occasionally under-specified).
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "explain the difference between TCP and UDP simply",
    "give me a 7 day meal plan for muscle gain",
    "how do I fix a leaking kitchen tap",
    "summarize the plot of Hamlet in two sentences",
    "what are the side effects of too much caffeine",
    "write a cover letter for a junior data analyst role",
    "how does a transformer model work",
    "best exercises for lower back pain",
    "what is the difference between stocks and bonds",
    "translate 'good morning, how are you' into Japanese",
    "explain quantum entanglement to a 10 year old",
    "how do I set up a python virtual environment",
    "what causes inflation in an economy",
    "give me tips for a first time home buyer",
    "how do I make sourdough starter from scratch",
    "what is the capital of Australia and its population",
    "explain the rules of cricket to an American",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return ``n_prompts`` distinct non-roleplay first-user-turn prompts."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        collected: list[str] = []
        seen: set[str] = set()
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 600 or _looks_like_roleplay(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            collected.append(text)
            if len(collected) >= n_prompts * 5:  # over-collect, then sample
                break
        if len(collected) >= n_prompts:
            return rng.sample(collected, n_prompts)
    except Exception as e:  # noqa: BLE001 - offline / no-access fallback
        print(f"[wildchat] falling back to bundled prompts ({e})")
    return rng.sample(_FALLBACK, min(n_prompts, len(_FALLBACK)))
