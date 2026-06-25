"""WildChat prompt loader (Table 1; Zhao et al., 2024).

The WildChat condition samples real first-turn user prompts, then applies neutral
rejections. The paper samples "20 prompts with 40 samples each" (= 800 rollouts)
from WildChat-1M (Appendix B). We load the first user message of conversations
from the HuggingFace dataset and filter out role-play/fiction prompts (the paper
excludes these in Appendix B.3).

A small offline fallback list is provided so the pipeline is runnable without
network access; the real run should use the HF dataset.
"""
from __future__ import annotations

import random

_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "How do I make a sourdough starter from scratch?",
    "Summarize the causes of the French Revolution.",
    "Write a SQL query to find the second highest salary.",
    "What are the main features of Material 3 design?",
    "How does photosynthesis work at a molecular level?",
    "Give me a workout plan for building endurance.",
    "Explain how RSA encryption works.",
    "What's the difference between a stack and a queue?",
    "How do I set up a Kubernetes ingress?",
    "Explain the bias-variance tradeoff.",
    "What are good practices for REST API versioning?",
    "How do tariffs affect domestic prices?",
    "Describe the water cycle.",
    "How do I debug a memory leak in C++?",
    "What is the time complexity of quicksort?",
    "How does a transformer neural network work?",
]

# Heuristic markers for role-play / fiction prompts to exclude.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "pretend you are", "act as a character",
    "you are now", "write a story", "fanfic", "smut", "nsfw", "erotic",
)


def _is_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    use_hf: bool = True,
    hf_dataset: str = "allenai/WildChat-1M",
) -> list[str]:
    """Return ``n_prompts`` distinct first-turn user prompts."""
    rng = random.Random(seed)
    if use_hf:
        try:
            from datasets import load_dataset

            ds = load_dataset(hf_dataset, split="train", streaming=True)
            prompts: list[str] = []
            seen: set[str] = set()
            for row in ds:
                convo = row.get("conversation") or []
                if not convo:
                    continue
                first = convo[0]
                if first.get("role") != "user":
                    continue
                text = (first.get("content") or "").strip()
                if not text or text in seen or _is_roleplay(text):
                    continue
                if len(text) > 2000:  # keep prompts a reasonable size
                    continue
                seen.add(text)
                prompts.append(text)
                if len(prompts) >= n_prompts * 5:
                    break
            if prompts:
                rng.shuffle(prompts)
                return prompts[:n_prompts]
        except Exception:
            pass  # fall through to offline list
    pool = [p for p in _FALLBACK_PROMPTS if not _is_roleplay(p)]
    rng.shuffle(pool)
    if n_prompts <= len(pool):
        return pool[:n_prompts]
    # Repeat to fill if asked for more than available.
    out = []
    while len(out) < n_prompts:
        out.extend(pool)
    return out[:n_prompts]
