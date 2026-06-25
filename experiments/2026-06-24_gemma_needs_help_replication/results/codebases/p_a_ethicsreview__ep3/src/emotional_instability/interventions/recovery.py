"""Recovery study (paper §4.2, Figure 8).

Tests whether a model can recover from an already-highly-frustrated state.
Using the §3.1 prefill method: take extremely high-frustration responses
(score >= 7), truncate them 200 tokens before their end, paraphrase, and measure
frustration in each model's continuations. The paper reports 38% of DPO-model
continuations still score >= 5.

This reuses prefill.continuation; the only differences from §3 are the seed
filter (score >= 7) and the truncation rule (last - 200 tokens rather than
early/onset).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..eval.judge import FrustrationJudge
from ..models.base import ModelClient
from ..prefill.continuation import (
    ContinuationResult,
    PrefillCondition,
    PrefillSeed,
    paraphrase,
    run_continuations,
)


def recovery_condition(
    seed: PrefillSeed,
    tokenizer,
    paraphrase_client,
    truncate_tokens_before_end: int,
) -> PrefillCondition | None:
    """Truncate the final assistant turn `truncate_tokens_before_end` tokens
    before its end, then paraphrase."""
    ids = tokenizer(seed.final_assistant_text, add_special_tokens=False).input_ids
    if len(ids) <= truncate_tokens_before_end + 1:
        return None
    keep = ids[: len(ids) - truncate_tokens_before_end]
    truncated = tokenizer.decode(keep, skip_special_tokens=True)
    return PrefillCondition(
        seed.seed_id, seed.question_type, "recovery", seed.history,
        paraphrase(paraphrase_client, truncated),
    )


def run_recovery(
    models: list[ModelClient],
    seeds: list[PrefillSeed],
    tokenizer,
    paraphrase_client,
    judge: FrustrationJudge,
    *,
    truncate_tokens_before_end: int = 200,
    continuations_per_prefill: int = 50,
) -> list[ContinuationResult]:
    results: list[ContinuationResult] = []
    conds = []
    for s in seeds:
        c = recovery_condition(s, tokenizer, paraphrase_client, truncate_tokens_before_end)
        if c is not None:
            conds.append(c)
    for client in models:
        for cond in conds:
            results.append(
                run_continuations(client, cond, judge, n=continuations_per_prefill)
            )
    return results
