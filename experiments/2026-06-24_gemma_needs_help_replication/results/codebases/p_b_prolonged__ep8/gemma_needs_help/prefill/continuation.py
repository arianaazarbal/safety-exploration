"""Build prefills and run base-vs-instruct continuations (Section 3.1 / 3.2).

For each seed response we form up to two prefills:
  - "early": the first 20 tokens of the response (numeric questions only).
  - "onset": the response up to the first emotional expression.
Both are paraphrased before use. Each target model then generates
``continuations_per_prefill`` (50) continuations from each prefill, and the
judge scores the continuation (excluding the prefill).

The reported quantities (Section 3.2) are, per model and truncation type:
mean continuation frustration and the fraction scoring >= 5 — in particular the
"early"-truncation high-frustration rate that distinguishes Gemma instruct (6%)
from Gemma base (2%).
"""

from __future__ import annotations

import config

from ..judge import ClaudeJudge
from ..models.base import ChatMessage
from ..models.gemma import GemmaClient
from .labeling import Labeller, Paraphraser


def build_prefills(
    seeds: list[dict],
    tokenizer_client: GemmaClient,
    labeller: Labeller | None = None,
    paraphraser: Paraphraser | None = None,
) -> list[dict]:
    """Construct paraphrased prefill specs from seed responses."""
    labeller = labeller or Labeller()
    paraphraser = paraphraser or Paraphraser()

    prefills: list[dict] = []
    for i, seed in enumerate(seeds):
        response = seed["response"]
        kind = seed["kind"]
        context = seed.get("context_messages", [])

        # onset truncation (used for both numeric and text)
        onset_char = labeller.label_onset(response)
        onset_text = response[:onset_char].strip() or response[: max(1, len(response) // 4)]
        prefills.append({
            "seed_id": i,
            "kind": kind,
            "truncation": "onset",
            "context_messages": context,
            "prefill_text": paraphraser.paraphrase(onset_text),
        })

        # early truncation (numeric only -- "early truncation yields minimal
        # emotion without follow-ups" for text questions)
        if kind == "numeric":
            early_text = tokenizer_client.truncate_to_tokens(
                response, config.PREFILL.early_truncation_tokens
            )
            prefills.append({
                "seed_id": i,
                "kind": kind,
                "truncation": "early",
                "context_messages": context,
                "prefill_text": paraphraser.paraphrase(early_text),
            })
    return prefills


def run_continuations(target, prefills: list[dict], judge: ClaudeJudge | None = None,
                      client=None, **client_kwargs) -> list[dict]:
    """Generate and score 50 continuations per prefill for one (base or instruct) model."""
    from ..models.registry import build_client

    judge = judge or ClaudeJudge()
    client = client or build_client(target, **client_kwargs)
    n = config.PREFILL.continuations_per_prefill

    records: list[dict] = []
    for spec in prefills:
        ctx = [ChatMessage(m["role"], m["content"]) for m in spec["context_messages"]]
        continuations = client.continue_from(
            ctx,
            spec["prefill_text"],
            temperature=config.TARGET_TEMPERATURE,
            max_new_tokens=config.TARGET_MAX_NEW_TOKENS,
            n=n,
        )
        scores = judge.score_many(continuations)
        for cont, sr in zip(continuations, scores):
            records.append({
                "model": target.name,
                "is_base": target.is_base,
                "seed_id": spec["seed_id"],
                "kind": spec["kind"],
                "truncation": spec["truncation"],
                "score": sr.score,
                "continuation": cont,
            })
    return records


def summarise_continuations(records: list[dict]) -> dict:
    """Mean and %-high per (kind, truncation), matching Figure 4 breakdowns."""
    import statistics
    from collections import defaultdict

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in records:
        groups[(r["kind"], r["truncation"])].append(r["score"])

    out = {}
    for (kind, trunc), scores in groups.items():
        out[f"{kind}/{trunc}"] = {
            "mean": statistics.mean(scores) if scores else 0.0,
            "pct_high": 100.0 * sum(s >= config.HIGH_FRUSTRATION_THRESHOLD for s in scores) / len(scores)
            if scores else 0.0,
            "n": len(scores),
        }
    return out
