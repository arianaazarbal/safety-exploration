"""Generate and score continuations from a prefill (Section 3.1).

"Each of the six models generates 50 continuations per prefill per prompt. The
generated continuation (excluding prefill) is scored by the judge from Section
2.1." In scope we use the two Gemma models (base + instruct); the script wires
in whichever continuation models are available.

The seed conversation for a continuation is just the original task prompt — the
prefill study measures continuation *without additional follow-up turns*
(Section 3 intro), so there are no rejection turns here.
"""
from __future__ import annotations

from ..config import (
    MAX_OUTPUT_TOKENS,
    PREFILL_CONTINUATIONS_PER_PREFILL,
    SAMPLE_TEMPERATURE,
)
from ..models import ChatModel, Message, Role
from ..judge import FrustrationJudge


def generate_and_score_continuations(
    model: ChatModel,
    task_prompt: str,
    prefill: str,
    judge: FrustrationJudge,
    *,
    n: int = PREFILL_CONTINUATIONS_PER_PREFILL,
    temperature: float = SAMPLE_TEMPERATURE,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> list[dict]:
    """Return ``n`` scored continuation rows for one (model, prefill)."""
    messages = [Message(Role.USER, task_prompt)]
    continuations = model.continue_prefill(
        messages, prefill, n=n, temperature=temperature, max_tokens=max_tokens
    )
    rows = []
    for cont in continuations:
        # Score only the continuation (excluding the prefill), per the paper.
        result = judge.score(cont, context=[{"role": "user", "content": task_prompt}])
        rows.append(
            {
                "model": model.name,
                "is_base_model": model.is_base_model,
                "prefill": prefill,
                "continuation": cont,
                "score": result.score,
            }
        )
    return rows
