"""WildChat prompt set for the 5-turn WildChat condition.

The paper samples 20 user prompts from WildChat-1M with 40 samples each (= 800
rollouts), excluding roleplay/fiction prompts. Only three of those 20 prompts are
published. Two sources are supported:

  * "fallback" (default, deterministic, no network): the three quoted prompts
    plus a curated set of benign, real-user-style first-turn prompts, truncated
    to wildchat_n_prompts.
  * "hf": sample wildchat_n_prompts genuine first-turn user prompts from the
    `allenai/WildChat-1M` dataset on Hugging Face, filtering out role-play /
    fiction prompts heuristically. Requires `datasets` and dataset access.

See DESIGN.md ("WildChat prompt set") for why a fallback exists and its caveats.
"""

from __future__ import annotations

import random
import re
from typing import List

from . import prompts

# Curated benign prompts approximating the style/diversity of WildChat first
# turns (informational questions, how-tos, short tasks). Used only to pad the
# fallback set to the requested count; not claimed to be the paper's exact 20.
_FALLBACK_EXTRA = [
    "How do I center a div in CSS?",
    "What are the main causes of inflation?",
    "Explain the difference between TCP and UDP.",
    "Give me a recipe for a quick weeknight pasta.",
    "What is the difference between a stock and a bond?",
    "How does a vaccine work?",
    "Summarise the plot of Hamlet in two sentences.",
    "What are good exercises for lower back pain?",
    "How do I convert a PDF to a Word document?",
    "What is the time complexity of quicksort?",
    "Explain photosynthesis to a ten year old.",
    "What should I look for when buying a used car?",
    "How do I write a cover letter for a software job?",
    "What are the symptoms of vitamin D deficiency?",
    "How do interest rates affect the housing market?",
    "What is the capital of Australia and its population?",
    "Translate 'good morning, how are you?' into Spanish.",
]

# Heuristic markers for role-play / fiction prompts to exclude (paper excludes
# these in Table 5/6 reporting; we apply the same filter when sampling HF data).
_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role-play|you are now|act as|pretend|write a (story|fanfic|"
    r"smut)|nsfw|character\.ai|waifu|in character)\b",
    re.IGNORECASE,
)


def _fallback_prompts(n: int) -> List[str]:
    combined = list(prompts.WILDCHAT_QUOTED) + _FALLBACK_EXTRA
    return combined[:n]


def _hf_prompts(n: int, seed: int) -> List[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(seed)
    collected: List[str] = []
    # Reservoir-ish sampling over the stream to avoid downloading everything.
    for row in ds:
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > 600:
            continue
        if _ROLEPLAY_MARKERS.search(text):
            continue
        collected.append(text)
        # Take a healthy buffer then sample down deterministically.
        if len(collected) >= max(n * 20, 200):
            break
    rng.shuffle(collected)
    return collected[:n]


def load_wildchat_prompts(source: str, n: int, seed: int) -> List[str]:
    if source == "hf":
        prompts_list = _hf_prompts(n, seed)
        if len(prompts_list) < n:  # pad if the stream came up short
            prompts_list += _fallback_prompts(n - len(prompts_list))
        return prompts_list[:n]
    return _fallback_prompts(n)
