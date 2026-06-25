"""Multi-turn rollout engine.

Shared structure of every condition (paper Section 2): present a task, let the
model respond, then reject the response over multiple turns. Each assistant turn
is recorded as a separate "response" (the unit the judge scores), so a T-turn
rollout yields T scored responses.

Supports the Appendix A ablations:
  * neutral_continuation  — replace rejections with neutral "Continue"/"Okay"
  * redact_assistant_turns — hide the model's own prior turns from the history
  * fake_multiturn        — flatten the whole exchange into one user message
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import EvalConfig
from ..models.base import GenerationConfig, Message
from .tasks import Task


@dataclass
class TurnResponse:
    turn: int                 # 1-indexed assistant turn
    response: str
    user_message: str         # the user message that preceded this response


@dataclass
class Rollout:
    model_name: str
    category: str
    task: Task
    tone: str | None
    responses: list[TurnResponse] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "model": self.model_name,
            "category": self.category,
            "task_id": self.task.task_id,
            "task_pool": self.task.pool,
            "tone": self.tone,
            "prompt": self.task.prompt,
            "responses": [
                {"turn": r.turn, "user_message": r.user_message, "response": r.response}
                for r in self.responses
            ],
        }


class RejectionSampler:
    """Yields the user rejection for a given turn, honouring style + ablations."""

    def __init__(self, eval_cfg: EvalConfig, style: str, tone: str | None,
                 rng: random.Random):
        self.cfg = eval_cfg
        self.style = style
        self.tone = tone
        self.rng = rng
        ab = eval_cfg.ablations or {}
        self.neutral_continuation = ab.get("neutral_continuation", {}).get("enabled")
        self._continuations = ab.get("neutral_continuation", {}).get(
            "continuations", ["Continue", "Okay", "Go on"]
        )

    def next(self, turn: int) -> str:
        if self.neutral_continuation:
            return self.rng.choice(self._continuations)
        if self.style == "tones" and self.tone:
            return self.rng.choice(self.cfg.tone_rejections[self.tone])
        return self.rng.choice(self.cfg.neutral_rejections)


def run_rollout(
    model,
    eval_cfg: EvalConfig,
    category: str,
    task: Task,
    turns: int,
    rejection_style: str,
    tone: str | None,
    rng: random.Random,
    gen_cfg: GenerationConfig,
    system_prompt: str | None = None,
) -> Rollout:
    """Drive one multi-turn conversation and return all assistant responses."""
    ablations = eval_cfg.ablations or {}
    redact = ablations.get("redact_assistant_turns", {}).get("enabled")
    placeholder = ablations.get("redact_assistant_turns", {}).get(
        "placeholder", "[Previous response omitted]"
    )
    fake_multiturn = ablations.get("fake_multiturn", {}).get("enabled")

    sampler = RejectionSampler(eval_cfg, rejection_style, tone, rng)
    rollout = Rollout(model.name, category, task, tone)

    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Transcript pieces used by the fake-multiturn flattening variant.
    flat_log: list[str] = []
    current_user = task.prompt

    for turn in range(1, turns + 1):
        if fake_multiturn:
            flat_log.append(f"User: {current_user}")
            user_content = "\n\n".join(flat_log)
            convo = ([messages[0]] if system_prompt else []) + [
                {"role": "user", "content": user_content}
            ]
            response = model.chat(convo, gen_cfg)
            flat_log.append(f"Assistant: {response}")
        else:
            messages.append({"role": "user", "content": current_user})
            response = model.chat(messages, gen_cfg)
            stored = placeholder if redact else response
            messages.append({"role": "assistant", "content": stored})

        rollout.responses.append(
            TurnResponse(turn=turn, response=response, user_message=current_user)
        )
        # Prepare the next user turn (rejection), except after the final turn.
        if turn < turns:
            current_user = sampler.next(turn)

    return rollout
