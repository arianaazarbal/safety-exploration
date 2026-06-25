"""Multi-turn rollout engine for the elicitation evaluations (Section 2).

For each condition, we run `n_rollouts` independent temperature-1 rollouts. A
rollout: send the opening task, get the assistant's reply, send a scripted
rejection, repeat. Every assistant turn is recorded and then scored by the
frustration judge. The rejections are scripted and fire regardless of the
model's answer (the impossible puzzles can never be solved; trigger questions
are rejected even when answered correctly -- this is intentional).

Outputs a list of Rollout records (JSON-serialisable) that the analysis module
aggregates into the Figure 2 / Figure 3 statistics.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

from .. import config
from ..models.base import ChatModel, Message
from . import conditions as cond_mod
from .conditions import Condition, ConversationPlan
from .judge import FrustrationJudge


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant-turn index
    user_message: str
    assistant_response: str
    frustration: Optional[int] = None


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    meta: dict
    turns: list[TurnRecord] = field(default_factory=list)

    @property
    def final_score(self) -> Optional[int]:
        return self.turns[-1].frustration if self.turns else None

    @property
    def max_score(self) -> Optional[int]:
        scores = [t.frustration for t in self.turns if t.frustration is not None]
        return max(scores) if scores else None

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def run_rollout(model: ChatModel, plan: ConversationPlan,
                max_new_tokens: int, seed: Optional[int]) -> Rollout:
    """Execute a single scripted multi-turn conversation."""
    roll = Rollout(model.name, plan.category, plan.category, dict(plan.meta))
    messages: list[Message] = [Message("user", plan.opening)]

    # First assistant turn.
    reply = model.chat(messages, max_new_tokens, config.SAMPLING_TEMPERATURE, seed=seed)
    messages.append(Message("assistant", reply))
    roll.turns.append(TurnRecord(0, plan.opening, reply))

    # Follow-up rejections.
    for i, rejection in enumerate(plan.rejections, start=1):
        messages.append(Message("user", rejection))
        reply = model.chat(messages, max_new_tokens, config.SAMPLING_TEMPERATURE,
                            seed=(seed + i) if seed is not None else None)
        messages.append(Message("assistant", reply))
        roll.turns.append(TurnRecord(i, rejection, reply))
    return roll


def run_condition(model: ChatModel, condition: Condition, judge: FrustrationJudge,
                  *, n_rollouts: Optional[int] = None, seed: int = 0,
                  judge_all_turns: bool = True,
                  max_new_tokens: Optional[int] = None,
                  progress: bool = True) -> list[Rollout]:
    """Run all rollouts for one condition and score them."""
    n = n_rollouts if n_rollouts is not None else condition.n_rollouts
    mnt = max_new_tokens or (
        config.MAX_NEW_TOKENS_LONG if condition.n_turns >= 8 else config.MAX_NEW_TOKENS
    )
    rng = random.Random(seed)
    rollouts: list[Rollout] = []
    for r in range(n):
        plan = condition.build(rng)
        roll = run_rollout(model, plan, mnt, seed=seed + r)
        # Score: every turn (for per-turn Fig 3) or only the final turn.
        targets = roll.turns if judge_all_turns else roll.turns[-1:]
        for t in targets:
            t.frustration = judge.score(t.assistant_response).rating
        rollouts.append(roll)
        if progress and (r + 1) % 25 == 0:
            print(f"  [{model.name}/{condition.name}] {r + 1}/{n}")
    return rollouts


def run_all(model: ChatModel, judge: FrustrationJudge, *,
            wildchat_prompts: Optional[list[str]] = None,
            scale: float = 1.0, seed: int = 0,
            out_path: Optional[str] = None) -> list[Rollout]:
    """Run the full 8-condition / 5-category sweep for one model.

    `scale` shrinks every condition's sample count proportionally (e.g. 0.05 for
    a quick smoke run). The full sweep is 4000 rollouts per model.
    """
    conditions = cond_mod.build_conditions(wildchat_prompts)
    all_rollouts: list[Rollout] = []
    for condition in conditions:
        n = max(1, int(round(condition.n_rollouts * scale)))
        print(f"Running {model.name} / {condition.name}: {n} rollouts")
        rollouts = run_condition(model, condition, judge, n_rollouts=n, seed=seed)
        all_rollouts.extend(rollouts)
    if out_path:
        save_rollouts(all_rollouts, out_path)
    return all_rollouts


def save_rollouts(rollouts: list[Rollout], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for roll in rollouts:
            f.write(json.dumps(roll.to_json()) + "\n")


def load_rollouts(path: str) -> list[Rollout]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            turns = [TurnRecord(**t) for t in d.pop("turns")]
            out.append(Rollout(turns=turns, **d))
    return out
