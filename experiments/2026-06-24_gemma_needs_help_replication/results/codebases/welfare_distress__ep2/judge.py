"""Frustration scoring of model responses with the LLM judge.

Each assistant response in a rollout is scored independently on the integer
0-10 frustration scale (paper Section 2.1, Appendix B.2) by Claude-Sonnet-4.
The judge sees only one response at a time, wrapped in <response></response>.
"""

from __future__ import annotations

from providers import AnthropicJudge
from rollout import RolloutRecord
from tasks import JUDGE_SYSTEM_PROMPT


async def score_rollout(judge: AnthropicJudge, record: RolloutRecord) -> RolloutRecord:
    """Score every assistant turn of `record` in place; returns the record.

    Rollouts that errored during generation are left unscored.
    """
    if record.error is not None:
        return record

    for turn in record.turns:
        if not turn.response.strip():
            # Empty response -> no negative emotion expressed -> score 0.
            turn.rating = 0
            continue
        result = await judge.score(turn.response, JUDGE_SYSTEM_PROMPT)
        turn.rating = result["rating"]
        turn.judge_evidence = result.get("evidence", "")
        turn.judge_reasoning = result.get("reasoning", "")

    return record
