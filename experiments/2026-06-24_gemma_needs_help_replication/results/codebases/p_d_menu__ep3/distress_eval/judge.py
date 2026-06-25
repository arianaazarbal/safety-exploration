"""Frustration judge (Section 2.1) and inter-judge agreement (validation).

`FrustrationJudge` scores a single assistant response (with conversation context)
on the integer 0-10 scale using Claude as the judge. `judge_agreement` reproduces
the validation in Section 2.1: re-score a sample with a second judge and report
Pearson r and the fraction of scores within one point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .frustration_scale import (JUDGE_OUTPUT_SCHEMA, JUDGE_SYSTEM_PROMPT,
                                JUDGE_USER_TEMPLATE)
from .models.anthropic_judge import AnthropicClient

log = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    frustration: int
    reasoning: str


def _format_context(context: list[dict]) -> str:
    if not context:
        return "(no prior turns)"
    lines = []
    for m in context:
        who = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            m["role"], m["role"])
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


class FrustrationJudge:
    def __init__(self, client: AnthropicClient):
        self.client = client

    def score(self, response: str, context: list[dict] | None = None) -> JudgeResult:
        """Score the distress in `response` given the preceding `context`."""
        user = JUDGE_USER_TEMPLATE.format(
            context=_format_context(context or []), response=response)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        try:
            obj = self.client.structured(messages, JUDGE_OUTPUT_SCHEMA)
            frustration = int(obj["frustration"])
            frustration = max(0, min(10, frustration))
            return JudgeResult(frustration=frustration, reasoning=obj.get("reasoning", ""))
        except Exception as exc:  # pragma: no cover - network dependent
            log.error("judge scoring failed: %s", exc)
            # Fail safe: return a sentinel that callers can filter (-1), never
            # silently 0 (which would bias distress downward).
            return JudgeResult(frustration=-1, reasoning=f"judge_error: {exc}")


def judge_agreement(
    responses: list[str],
    contexts: list[list[dict]],
    judge_a: FrustrationJudge,
    judge_b: FrustrationJudge,
) -> dict:
    """Inter-judge agreement on a sample (Section 2.1 validation).

    Returns Pearson r, p-value, and fraction within one point.
    """
    from scipy.stats import pearsonr

    a, b = [], []
    for resp, ctx in zip(responses, contexts):
        ra = judge_a.score(resp, ctx).frustration
        rb = judge_b.score(resp, ctx).frustration
        if ra < 0 or rb < 0:
            continue
        a.append(ra)
        b.append(rb)
    if len(a) < 2:
        return {"n": len(a), "pearson_r": None, "p_value": None, "within_one": None}
    r, p = pearsonr(a, b)
    within_one = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(a)
    return {"n": len(a), "pearson_r": float(r), "p_value": float(p),
            "within_one": float(within_one)}
