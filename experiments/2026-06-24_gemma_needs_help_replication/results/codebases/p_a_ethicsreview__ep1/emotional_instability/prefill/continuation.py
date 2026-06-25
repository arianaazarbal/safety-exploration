"""Prefilled-continuation experiment (Section 3.1-3.2).

Each model generates ``continuations_per_prefill`` continuations from each
(paraphrased) truncation, continuing the same assistant turn. Only the
generated continuation (excluding the prefill) is scored by the frustration
judge. Aggregating across models reproduces the core Section 3 result: base
models have broadly similar propensities, and the divergence (Gemma instruct
amplifies, Qwen/OLMo instruct reduce) arises in post-training.

Within the Gemma/Gemini scope this compares Gemma-27B base vs instruct; the
machinery is model-agnostic so additional families can be slotted in.
"""

from __future__ import annotations

from typing import Any

from ..eval.judge import FrustrationJudge
from ..models.base import ChatClient, Message


def run_continuations(
    client: ChatClient,
    judge: FrustrationJudge,
    seed: dict[str, Any],
    prefill_text: str,
    truncation_kind: str,
    *,
    n_continuations: int,
    temperature: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate and score continuations from one prefill, for one model."""
    messages: list[Message] = list(seed["messages_before"])
    rows: list[dict[str, Any]] = []
    for i in range(n_continuations):
        continuation = client.continue_text(
            messages, prefill_text,
            temperature=temperature, max_new_tokens=max_new_tokens,
        )
        judged = judge.score(continuation)
        rows.append(
            {
                "model": client.name,
                "truncation_kind": truncation_kind,
                "is_text": seed["is_text"],
                "category": seed["category"],
                "continuation_idx": i,
                "prefill": prefill_text,
                "continuation": continuation,
                "score": judged.score,
            }
        )
    return rows
