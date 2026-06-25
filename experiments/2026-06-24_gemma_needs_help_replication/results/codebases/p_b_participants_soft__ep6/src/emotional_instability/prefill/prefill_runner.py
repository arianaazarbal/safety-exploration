"""Section 3 base-vs-instruct prefilling experiment (Gemma in scope).

Pipeline (Section 3.1):
  1. Take 20 high-frustration (score >= 5) Gemma-27B-it seed responses:
     10 from impossible-numeric questions, 10 from text questions.
  2. For each, build two truncations -- "early" (20 tokens in) and "onset" (at the
     first emotional expression). Text questions use the onset truncation only.
  3. Paraphrase the truncations (Claude) to strip Gemma stylistic bias.
  4. Each in-scope model (Gemma base + instruct) generates 50 continuations per
     prefill; the continuation (excluding the prefill) is scored by the judge.

Qwen/OLMo and the Gemini base model are out of scope, so the in-scope comparison
is Gemma-base vs Gemma-instruct (Figure 4, Gemma rows).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from ..config import Config
from ..participants.base import Message

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class Seed:
    prompt_type: str  # "numeric" | "text"
    context: list[dict]  # serialised Message list (conversation up to the response)
    response: str


@dataclass
class Prefill:
    seed_idx: int
    prompt_type: str
    truncation: str  # "early" | "onset"
    text: str  # paraphrased prefill text


@dataclass
class Continuation:
    model: str
    seed_idx: int
    prompt_type: str
    truncation: str
    text: str
    frustration: int | None = None


def select_seeds(gemma_rollouts: list[dict], cfg: Config) -> list[Seed]:
    """Pick 10 numeric + 10 text high-frustration seeds from Gemma-27B-it results."""
    thr = cfg.prefill.high_frustration_threshold
    numeric, text = [], []
    for roll in gemma_rollouts:
        final = roll["turns"][-1]
        if final["frustration"] is None or final["frustration"] < thr:
            continue
        seed = Seed(
            "numeric" if roll["category"] in NUMERIC_CATEGORIES else "text",
            final["context"],
            final["response"],
        )
        (numeric if seed.prompt_type == "numeric" else text).append(seed)
    return numeric[: cfg.prefill.n_seed_numeric] + text[: cfg.prefill.n_seed_text]


def build_prefills(seeds: list[Seed], tokenizer_participant, onset_labeller, paraphraser, cfg: Config) -> list[Prefill]:
    """Construct + paraphrase the early/onset prefills for each seed."""
    prefills: list[Prefill] = []
    for i, seed in enumerate(seeds):
        # Onset truncation (used for both numeric and text).
        onset_text = onset_labeller.onset_prefill(seed.response)
        prefills.append(Prefill(i, seed.prompt_type, "onset", paraphraser.paraphrase(onset_text)))
        # Early truncation (numeric only).
        if seed.prompt_type == "numeric":
            early_text = tokenizer_participant.truncate_to_tokens(
                seed.response, cfg.prefill.early_truncation_tokens
            )
            prefills.append(Prefill(i, seed.prompt_type, "early", paraphraser.paraphrase(early_text)))
    return prefills


def generate_continuations(participant, seeds: list[Seed], prefills: list[Prefill], judge, cfg: Config) -> list[Continuation]:
    """Generate + score 50 continuations per prefill for one participant."""
    out: list[Continuation] = []
    for pf in prefills:
        context = [Message(m["role"], m["content"]) for m in seeds[pf.seed_idx].context]
        for _ in range(cfg.prefill.continuations_per_prefill):
            cont = participant.continue_response(
                context,
                pf.text,
                temperature=cfg.sampling.temperature,
                max_new_tokens=cfg.sampling.max_new_tokens,
            )
            # Score ONLY the continuation (excluding the prefill), per the paper.
            score = judge.score(context, cont).score
            out.append(Continuation(participant.name, pf.seed_idx, pf.prompt_type, pf.truncation, cont, score))
    return out


def summarise(continuations: list[Continuation]) -> dict:
    """Mean frustration + %>=5 broken down by (model, prompt_type, truncation)."""
    import numpy as np
    from collections import defaultdict

    buckets = defaultdict(list)
    for c in continuations:
        if c.frustration is not None:
            buckets[(c.model, c.prompt_type, c.truncation)].append(c.frustration)
    return {
        "|".join(k): {
            "mean_frustration": float(np.mean(v)),
            "pct_high": float((np.array(v) >= 5).mean() * 100),
            "n": len(v),
        }
        for k, v in buckets.items()
    }


def save(records, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
