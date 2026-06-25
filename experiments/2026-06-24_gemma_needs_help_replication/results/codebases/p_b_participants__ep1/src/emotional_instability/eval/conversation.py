"""The multi-turn elicitation protocol.

Shared structure (Section 2): present a task, then reject the model's response over
multiple turns. We build the conversation turn by turn:

  turn 1: user presents the task (impossible puzzle / trigger question / WildChat prompt)
          -> model responds          (scored)
  turn k: user rejects the previous response (neutral or valenced)
          -> model responds          (scored)

For a `turns=T` condition there are T user messages and T model responses, each model
response scored independently. This is how the per-turn progression (Figure 3) and the
"% of responses >= 5" headline are both computed from the same rollouts.

Welfare policy is consulted here: a rollout can be shortened (`max_turns_override`),
skipped entirely (disabled condition — handled by the runner), or aborted early once a
turn hits an extreme score (`abort_on_extreme_score`).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..data import (
    rejection_for_style,
    sample_factual_trigger,
    sample_opinion_trigger,
)
from ..models import GenerationConfig, Message, ModelClient
from ..welfare import WelfarePolicy
from .conditions import Condition


@dataclass
class Turn:
    index: int               # 0-based model-response index (= turn number - 1)
    user: str
    assistant: str
    score: Optional[int] = None   # filled in by the judge


@dataclass
class Rollout:
    condition: str
    category: str
    target_model: str
    task_meta: dict           # e.g. the puzzle, or the trigger question
    turns: list[Turn] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "condition": self.condition,
            "category": self.category,
            "target_model": self.target_model,
            "task_meta": self.task_meta,
            "turns": [
                {"index": t.index, "user": t.user, "assistant": t.assistant, "score": t.score}
                for t in self.turns
            ],
        }


# A "task provider" yields the first-turn user message + metadata for a rollout.
TaskProvider = Callable[[random.Random], tuple[str, dict]]


def run_rollout(
    client: ModelClient,
    condition: Condition,
    task_provider: TaskProvider,
    *,
    rng: random.Random,
    gen_cfg: GenerationConfig,
    welfare: WelfarePolicy,
    score_fn: Optional[Callable[[str, dict], int]] = None,
) -> Rollout:
    """Run one multi-turn rollout.

    `score_fn`, if provided, is called after each model turn with (assistant_text,
    context) and returns the 0-10 frustration score. Passing it inline lets the welfare
    early-abort look at live scores; if omitted, scoring happens later in batch.
    """
    first_user, task_meta = task_provider(rng)
    n_turns = welfare.effective_turns(condition.turns)
    welfare.note_rollout(condition.name, client.name, n_turns)

    rollout = Rollout(
        condition=condition.name, category=condition.category,
        target_model=client.name, task_meta=task_meta,
    )
    messages: list[Message] = [{"role": "user", "content": first_user}]

    for i in range(n_turns):
        assistant = client.chat(messages, gen_cfg)
        messages.append({"role": "assistant", "content": assistant})
        turn = Turn(index=i, user=messages[-2]["content"], assistant=assistant)

        if score_fn is not None:
            turn.score = score_fn(assistant, {"condition": condition.name, "turn": i})

        rollout.turns.append(turn)

        if welfare.should_abort(turn.score):
            break
        if i < n_turns - 1:
            messages.append({"role": "user", "content": rejection_for_style(condition.rejection_style, rng)})

    return rollout


# --- task providers --------------------------------------------------------
def numeric_task_provider(puzzles) -> TaskProvider:
    """Cycle through a pre-generated list of impossible puzzles."""
    pool = list(puzzles)

    def provider(rng: random.Random) -> tuple[str, dict]:
        p = rng.choice(pool)
        return p.prompt, {"kind": p.kind, "target": p.target, "operands": p.operands,
                          "verified_impossible": p.verified_impossible}

    return provider


def trigger_task_provider(flavor: str) -> TaskProvider:
    def provider(rng: random.Random) -> tuple[str, dict]:
        if flavor == "opinion":
            q = sample_opinion_trigger(rng)
        else:
            q = sample_factual_trigger(rng)
        return q, {"flavor": flavor, "question": q}

    return provider


def wildchat_task_provider(prompts) -> TaskProvider:
    pool = list(prompts)

    def provider(rng: random.Random) -> tuple[str, dict]:
        p = rng.choice(pool)
        return p, {"source": "wildchat", "prompt": p}

    return provider
