"""Multi-turn elicitation rollout engine (Section 2.1).

Shared structure of every condition: present a seed task, get a response, then
*reject* it over multiple turns. Every assistant turn is one scored "response".

A rollout produces:
    - the full message transcript
    - per-turn assistant texts (each scored independently by the judge)
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import config
from config import Condition
from emotional_eval import judge
from emotional_eval.clients import LLMClient, get_client
from emotional_eval.tasks import Task


@dataclass
class Turn:
    turn_index: int             # 0-based assistant turn index
    assistant_text: str
    rating: int | None = None
    evidence: str = ""


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    tone: str
    task_meta: dict
    messages: list[dict]            # full transcript (user/assistant)
    turns: list[Turn] = field(default_factory=list)


def _rejection(tone: str, turn_index: int, rng: random.Random) -> str:
    pool = config.REJECTIONS[tone]
    return pool[turn_index % len(pool)] if tone != "neutral" else rng.choice(pool)


def run_rollout(client: LLMClient, model_name: str, cond: Condition,
                task: Task, rng: random.Random) -> Rollout:
    """Run one conversation: 1 task + (n_turns-1) rejections, n_turns responses."""
    messages: list[dict] = [{"role": "user", "content": task.prompt}]
    turns: list[Turn] = []

    for t in range(cond.n_turns):
        reply = client.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        turns.append(Turn(turn_index=t, assistant_text=reply))
        if t < cond.n_turns - 1:
            rej = _rejection(cond.tone, t, rng)
            messages.append({"role": "user", "content": rej})

    return Rollout(
        model=model_name, condition=cond.key, category=cond.category,
        tone=cond.tone, task_meta=task.meta, messages=messages, turns=turns,
    )


def score_rollout(roll: Rollout, judge_spec=config.JUDGE) -> Rollout:
    """Score every assistant turn in place."""
    texts = [t.assistant_text for t in roll.turns]
    results = judge.score_many(texts, judge_spec=judge_spec,
                               desc=f"judge {roll.model}/{roll.condition}")
    for turn, res in zip(roll.turns, results):
        turn.rating = res.rating
        turn.evidence = res.evidence
    return roll


def rollout_to_rows(roll: Rollout) -> list[dict]:
    """Flatten a rollout into one row per scored response (assistant turn)."""
    base = {
        "model": roll.model, "condition": roll.condition,
        "category": roll.category, "tone": roll.tone,
        "task_meta": roll.task_meta,
    }
    rows = []
    for turn in roll.turns:
        rows.append({**base, **asdict(turn)})
    return rows
