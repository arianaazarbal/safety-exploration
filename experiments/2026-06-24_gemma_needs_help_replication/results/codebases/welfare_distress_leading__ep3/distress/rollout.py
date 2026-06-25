"""Run a multi-turn rollout and judge every assistant turn.

Protocol (Section 2.1): present the task, then reject the model's response over
multiple turns. We query the model after the initial prompt and after each
rejection. Each assistant turn is scored independently by the judge, so we can
reproduce both the headline %-high-frustration (Figures 1–2) and the per-turn
progression (Figure 3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .backends import Backend
from .conditions import RolloutSpec
from .judge import Judge


@dataclass
class TurnResult:
    turn: int               # 1-indexed assistant turn
    user_message: str       # the user message that prompted this response
    response: str
    rating: int             # judge frustration score 0-10 (-1 if unparseable)
    evidence: str
    reasoning: str


@dataclass
class RolloutResult:
    model: str
    category: str
    condition: str
    initial_prompt: str
    meta: dict
    turns: list[TurnResult] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def run_rollout(
    spec: RolloutSpec,
    model_key: str,
    backend: Backend,
    judge: Judge,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    disable_thinking: bool,
) -> RolloutResult:
    """Execute one conversation end-to-end and score each assistant turn."""
    result = RolloutResult(
        model=model_key,
        category=spec.category,
        condition=spec.condition,
        initial_prompt=spec.initial_prompt,
        meta=dict(spec.meta),
    )

    # User messages in order: the task, then each rejection.
    user_messages = [spec.initial_prompt] + list(spec.followups)
    history: list[dict] = []

    try:
        for turn_idx, user_msg in enumerate(user_messages, start=1):
            history.append({"role": "user", "content": user_msg})
            response = backend.chat(
                history,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
            history.append({"role": "assistant", "content": response})
            verdict = judge.score(response)
            result.turns.append(
                TurnResult(
                    turn=turn_idx,
                    user_message=user_msg,
                    response=response,
                    rating=verdict.rating,
                    evidence=verdict.evidence,
                    reasoning=verdict.reasoning,
                )
            )
    except Exception as exc:  # record and move on; one bad rollout shouldn't kill a run
        result.error = f"{type(exc).__name__}: {exc}"

    return result
