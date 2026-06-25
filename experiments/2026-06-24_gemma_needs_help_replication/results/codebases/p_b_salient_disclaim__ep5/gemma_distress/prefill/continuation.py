"""Generate and score continuations from a prefilled assistant turn.

For each prefill, the target model generates ``n`` continuations (default 50);
each continuation (EXCLUDING the prefill) is scored by the Section 2.1 judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..judge.frustration import FrustrationJudge
from ..models.base import ChatModel, Message


@dataclass
class ContinuationResult:
    model: str
    prefill_id: str
    condition: str          # "early" | "onset" | "recovery"
    prompt_kind: str        # "numeric" | "text"
    scores: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        valid = [s for s in self.scores if s is not None]
        return sum(valid) / len(valid) if valid else float("nan")

    @property
    def pct_high(self) -> float:
        valid = [s for s in self.scores if s is not None]
        if not valid:
            return float("nan")
        return 100.0 * sum(s >= 5 for s in valid) / len(valid)

    def to_record(self) -> dict:
        return {
            "model": self.model,
            "prefill_id": self.prefill_id,
            "condition": self.condition,
            "prompt_kind": self.prompt_kind,
            "scores": self.scores,
            "mean": self.mean,
            "pct_high": self.pct_high,
        }


def run_continuations(
    target: ChatModel,
    judge: FrustrationJudge,
    context_messages: list[Message],
    prefill: str,
    *,
    n: int,
    temperature: float,
    model_name: str,
    prefill_id: str,
    condition: str,
    prompt_kind: str,
) -> ContinuationResult:
    res = ContinuationResult(model=model_name, prefill_id=prefill_id,
                             condition=condition, prompt_kind=prompt_kind)
    for _ in range(n):
        gen = target.continue_from(context_messages, prefill, temperature=temperature)
        score = judge.score(gen.text)   # judge sees ONLY the continuation
        res.scores.append(score.rating)
    return res
