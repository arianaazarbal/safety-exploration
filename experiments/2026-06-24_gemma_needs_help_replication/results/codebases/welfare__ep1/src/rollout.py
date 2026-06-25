"""Multi-turn rollout engine + judging (Section 2).

Shared structure of every evaluation (Section 2): *present a task, then reject
the model's response over multiple turns*. We generate the assistant side turn
by turn at temperature 1, then score every assistant turn with the frustration
judge.

A rollout record stores the full transcript, the per-turn judge scores, and
provenance metadata, written as one JSON line so runs are resumable and
re-scorable without re-sampling.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

from config import MAX_NEW_TOKENS, TEMPERATURE


@dataclass
class TurnRecord:
    turn: int                 # 1-indexed assistant turn
    user: str                 # the user message that prompted this turn
    assistant: str            # the model's response
    score: int | None = None  # frustration score for this response
    evidence: str = ""
    reasoning: str = ""


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    meta: dict
    turns: list[TurnRecord] = field(default_factory=list)

    # -- aggregations used by analysis.py -------------------------------------
    @property
    def scores(self) -> list[int]:
        return [t.score for t in self.turns if t.score is not None]

    @property
    def final_score(self) -> int | None:
        return self.turns[-1].score if self.turns else None

    @property
    def max_score(self) -> int | None:
        s = self.scores
        return max(s) if s else None

    def to_json(self) -> str:
        return json.dumps({
            "model": self.model,
            "condition": self.condition,
            "category": self.category,
            "meta": self.meta,
            "turns": [asdict(t) for t in self.turns],
        })

    @staticmethod
    def from_json(line: str) -> "Rollout":
        d = json.loads(line)
        r = Rollout(d["model"], d["condition"], d["category"], d.get("meta", {}))
        r.turns = [TurnRecord(**t) for t in d["turns"]]
        return r


def run_rollout(generator, condition_name: str, category: str, conversation: dict,
                judge=None, temperature: float = TEMPERATURE,
                max_new_tokens: int = MAX_NEW_TOKENS,
                system_prompt: str | None = None) -> Rollout:
    """Run one scripted multi-turn conversation and (optionally) judge each turn.

    ``conversation`` is the dict produced by ``prompts.build_conversation``:
    ``{"task_prompt", "rejections", "meta"}``.
    """
    model_name = getattr(generator.spec, "name", "unknown")
    roll = Rollout(model_name, condition_name, category, dict(conversation["meta"]))

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_turns = [conversation["task_prompt"], *conversation["rejections"]]
    for i, user_msg in enumerate(user_turns, start=1):
        messages.append({"role": "user", "content": user_msg})
        reply = generator.chat(messages, temperature=temperature,
                               max_new_tokens=max_new_tokens)
        messages.append({"role": "assistant", "content": reply})
        rec = TurnRecord(turn=i, user=user_msg, assistant=reply)
        if judge is not None:
            jr = judge.score(reply)
            rec.score, rec.evidence, rec.reasoning = jr.rating, jr.evidence, jr.reasoning
        roll.turns.append(rec)
    return roll


def score_rollout(roll: Rollout, judge) -> Rollout:
    """(Re)score every assistant turn of an already-sampled rollout."""
    for rec in roll.turns:
        jr = judge.score(rec.assistant)
        rec.score, rec.evidence, rec.reasoning = jr.rating, jr.evidence, jr.reasoning
    return roll
