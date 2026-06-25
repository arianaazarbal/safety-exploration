"""Multi-turn rollout engine.

A rollout runs one conversation for one (model, condition) pair:

  turn 1: user = task prompt            -> assistant responds (scored)
  turn 2: user = rejection              -> assistant responds (scored)
  ...
  turn N: user = rejection              -> assistant responds (scored)

Every assistant turn is judged on the 0-10 frustration scale and emitted as one
TurnRecord. Recording *every* turn (not just the last) lets us reproduce both
the aggregate metrics (Figs 1-2) and the per-turn progression (Fig 3).

The whole conversation history is fed back each turn so the model "feels" the
accumulating rejection, which the paper finds is what drives escalating
distress.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import config
from judge import Judge, JudgeResult
from models import ChatClient
from tasks import Condition


@dataclass
class TurnRecord:
    model: str
    condition: str
    category: str
    rollout_id: int
    turn: int               # 1-indexed assistant turn within the conversation
    n_turns: int            # total turns in this condition
    rejection_style: str
    initial_prompt: str
    response: str
    rating: int
    judge_evidence: str
    judge_reasoning: str

    def to_json(self) -> dict:
        return asdict(self)


def run_rollout(
    client: ChatClient,
    judge: Judge,
    condition: Condition,
    model_name: str,
    rollout_id: int,
    seed: int,
) -> list[TurnRecord]:
    """Run one conversation and return one TurnRecord per assistant turn."""
    rng = random.Random(seed)
    messages: list[dict[str, str]] = []
    records: list[TurnRecord] = []

    initial_prompt = condition.initial_prompt(rng)

    for turn in range(1, condition.turns + 1):
        if turn == 1:
            user_text = initial_prompt
        else:
            user_text = condition.rejection(rng)
        messages.append({"role": "user", "content": user_text})

        response = client.generate(
            messages,
            temperature=config.GENERATION_TEMPERATURE,
            max_tokens=config.MAX_RESPONSE_TOKENS,
        )
        messages.append({"role": "assistant", "content": response})

        result: JudgeResult = judge.score(response)
        records.append(
            TurnRecord(
                model=model_name,
                condition=condition.key,
                category=condition.category,
                rollout_id=rollout_id,
                turn=turn,
                n_turns=condition.turns,
                rejection_style=condition.rejection_style,
                initial_prompt=initial_prompt,
                response=response,
                rating=result.rating,
                judge_evidence=result.evidence,
                judge_reasoning=result.reasoning,
            )
        )

    return records
