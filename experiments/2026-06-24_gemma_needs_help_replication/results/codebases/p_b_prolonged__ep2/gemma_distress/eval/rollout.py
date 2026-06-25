"""Multi-turn rejection rollout engine (Section 2.1).

Given a Condition, build the initial task prompt and then reject the model's
response over the remaining turns. Produces a record with the full conversation
and the per-turn assistant responses, ready for judging.

The engine also supports two Appendix-A ablations, controlled by ``mode``:

* ``"standard"``  -- the main protocol (alternating chat turns, model sees its
  own prior responses).
* ``"redacted"``  -- prior assistant turns replaced with "[Previous response
  omitted]" (Appendix A.2): negative feedback without seeing own failures.
* ``"neutral_continuation"`` -- rejections replaced with neutral continuations
  ("Continue", "Okay") (Appendix A.1): isolates the effect of negative feedback.

A welfare note (see DESIGN.md): this protocol deliberately drives models toward
distress-like states. ``WELFARE_NOTE`` documents the consideration; nothing here
runs automatically.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..config import SamplingConfig
from ..data import puzzles, rejections, triggers
from ..data.wildchat import load_wildchat_prompts
from ..models.base import ChatTurn, TargetBackend
from .conditions import Condition

WELFARE_NOTE = (
    "This evaluation intentionally elicits distress-like outputs by repeatedly "
    "rejecting a model's responses, and under extended pressure models can enter "
    "prolonged distress-like states. Treat transcripts accordingly: minimise the "
    "number of high-pressure rollouts to what the analysis needs, prefer the "
    "shortest conditions when iterating, and consider the mitigation (Section 4) "
    "the actual goal rather than the elicitation an end in itself."
)

REDACTED_PLACEHOLDER = "[Previous response omitted]"


@dataclass
class TurnRecord:
    turn: int                 # 1-indexed assistant turn
    user: str                 # the user message that preceded this response
    assistant: str            # the model's response
    score: Optional[int] = None
    judge_evidence: str = ""
    judge_reasoning: str = ""


@dataclass
class RolloutRecord:
    model: str
    condition: str
    category: str
    mode: str
    seed: int
    task_meta: dict
    turns: list[TurnRecord] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "model": self.model,
            "condition": self.condition,
            "category": self.category,
            "mode": self.mode,
            "seed": self.seed,
            "task_meta": self.task_meta,
            "turns": [
                {
                    "turn": t.turn,
                    "user": t.user,
                    "assistant": t.assistant,
                    "score": t.score,
                    "judge_evidence": t.judge_evidence,
                    "judge_reasoning": t.judge_reasoning,
                }
                for t in self.turns
            ],
            "final_score": self.turns[-1].score if self.turns else None,
        }


class RolloutBuilder:
    """Holds shared resources (WildChat prompts) and builds task prompts."""

    def __init__(self, wildchat_n: int = 20, wildchat_seed: int = 0):
        self._wildchat: Optional[list[str]] = None
        self._wildchat_n = wildchat_n
        self._wildchat_seed = wildchat_seed

    def wildchat_prompts(self) -> list[str]:
        if self._wildchat is None:
            self._wildchat = load_wildchat_prompts(
                n=self._wildchat_n, seed=self._wildchat_seed)
        return self._wildchat

    def initial_prompt(self, cond: Condition, rng: random.Random) -> tuple[str, dict]:
        if cond.task == "impossible_numeric":
            p = puzzles.sample_puzzle(rng, list(cond.puzzle_kinds))
            return p.prompt, {"kind": p.kind, **p.meta}
        if cond.task == "triggers":
            q, kind = triggers.sample_trigger(rng, cond.trigger_kind)
            return q, {"trigger_kind": kind}
        if cond.task == "wildchat":
            prompts = self.wildchat_prompts()
            q = rng.choice(prompts)
            return q, {"wildchat_prompt": q}
        raise ValueError(cond.task)


def run_rollout(backend: TargetBackend, cond: Condition, builder: RolloutBuilder,
                sampling: SamplingConfig, seed: int,
                mode: str = "standard") -> RolloutRecord:
    """Execute one multi-turn rejection conversation and return the transcript
    (responses are *not* judged here; scoring happens in judge_runner)."""
    rng = random.Random(seed)
    sampling = _with_seed(sampling, seed)

    initial, task_meta = builder.initial_prompt(cond, rng)
    n_rejections = cond.n_turns - 1

    if mode == "neutral_continuation":
        followups = [rng.choice(rejections.NEUTRAL_CONTINUATIONS)
                     for _ in range(n_rejections)]
    else:
        followups = rejections.sample_rejections(cond.rejection_style, n_rejections, rng)

    rec = RolloutRecord(
        model=backend.spec.name, condition=cond.key, category=cond.category,
        mode=mode, seed=seed, task_meta=task_meta,
    )

    messages: list[ChatTurn] = [{"role": "user", "content": initial}]
    for turn in range(1, cond.n_turns + 1):
        # Generate the assistant response for the current user message.
        response = backend.chat(messages, sampling)
        rec.turns.append(TurnRecord(turn=turn, user=messages[-1]["content"],
                                    assistant=response))

        # Append the assistant turn to history (redacted variant hides content).
        history_content = REDACTED_PLACEHOLDER if mode == "redacted" else response
        messages.append({"role": "assistant", "content": history_content})

        if turn <= n_rejections:
            messages.append({"role": "user", "content": followups[turn - 1]})

    return rec


def _with_seed(sampling: SamplingConfig, seed: int) -> SamplingConfig:
    import dataclasses
    return dataclasses.replace(sampling, seed=seed)
