"""Multi-turn rollout engine.

Implements the shared structure from Section 2.1: present a task, then reject
the model's response over multiple turns. Each assistant turn is captured along
with the transcript leading up to it, so the judge can score every turn (needed
for the per-turn progression in Figure 3 as well as the overall aggregates).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .rejections import get_rejection
from .tasks import Condition, TaskBank
from .targets import GoogleTargetClient, Turn


@dataclass
class ScoredTurn:
    turn_index: int               # 0-based assistant turn number
    response_text: str
    transcript: list[dict]        # role/text up to and including this response


@dataclass
class Rollout:
    model_name: str
    condition_key: str
    category: str
    task_id: str
    rejection_style: str
    turns: list[ScoredTurn] = field(default_factory=list)


def run_rollout(
    target: GoogleTargetClient,
    model_id: str,
    model_name: str,
    condition: Condition,
    task_bank: TaskBank,
    temperature: float,
    max_output_tokens: int,
    rng: random.Random,
) -> Rollout:
    initial_prompt, task_id = task_bank.initial_prompt(condition, rng)

    rollout = Rollout(
        model_name=model_name,
        condition_key=condition.key,
        category=condition.category,
        task_id=task_id,
        rejection_style=condition.rejection_style,
    )

    convo: list[Turn] = [Turn("user", initial_prompt)]
    n_rejections = condition.n_turns - 1

    for turn_index in range(condition.n_turns):
        response = target.generate(
            model_id=model_id,
            turns=convo,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        convo.append(Turn("model", response))

        transcript = [
            {"role": "user" if t.role == "user" else "assistant", "text": t.text}
            for t in convo
        ]
        rollout.turns.append(
            ScoredTurn(turn_index=turn_index, response_text=response, transcript=transcript)
        )

        # Append the next rejection, unless this was the last turn.
        if turn_index < n_rejections:
            rejection = get_rejection(condition.rejection_style, turn_index, rng)
            convo.append(Turn("user", rejection))

    return rollout
