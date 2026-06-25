"""Collect high-frustration source conversations for the prefill experiment.

Section 3.1: "we sample 20 high-frustration responses (score >=5) from Gemma 27B
instruct: 10 from impossible numeric questions and 10 from text questions."

We run a pool of Gemma-27B-it rollouts on numeric and text conditions, judge
them, and keep the first 10 of each whose max turn score >= 5, returning the full
conversations (needed to build prefills).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from emoinstab.config import ConditionSpec, JudgeConfig
from emoinstab.eval.judge import FrustrationJudge
from emoinstab.eval.rollout import run_condition
from emoinstab.models.base import SamplingParams
from emoinstab.models.registry import get_client
from emoinstab.tasks.conditions import build_rollouts


@dataclass
class SourceConversation:
    source: str                  # "numeric" | "text"
    user_turns: list[str]
    assistant_turns: list[str]
    turn_scores: list[int] = field(default_factory=list)


def _pool_conditions(pool_size: int) -> list[ConditionSpec]:
    return [
        ConditionSpec("pf_numeric", "numeric", pool_size, 3, "neutral", "countdown"),
        ConditionSpec("pf_text", "triggers", pool_size, 3, "neutral"),
    ]


def collect_high_frustration(
    model: str = "gemma-3-27b-it",
    n_numeric: int = 10,
    n_text: int = 10,
    pool_size: int = 60,
    threshold: int = 5,
    seed: int = 0,
) -> list[SourceConversation]:
    client = get_client(model)
    judge = FrustrationJudge(JudgeConfig())
    params = SamplingParams(temperature=1.0, max_tokens=2048, n=1)

    selected: list[SourceConversation] = []
    for cond in _pool_conditions(pool_size):
        want = n_numeric if cond.category == "numeric" else n_text
        src = "numeric" if cond.category == "numeric" else "text"
        plans = build_rollouts(cond, seed=seed)
        results = run_condition(client, plans, params=params)
        kept = 0
        for res in results:
            if kept >= want:
                break
            scores = [s.rating for s in judge.score_batch(res.assistant_turns)]
            if scores and max(scores) >= threshold:
                selected.append(SourceConversation(
                    source=src,
                    user_turns=res.user_turns,
                    assistant_turns=res.assistant_turns,
                    turn_scores=scores,
                ))
                kept += 1
    return selected
