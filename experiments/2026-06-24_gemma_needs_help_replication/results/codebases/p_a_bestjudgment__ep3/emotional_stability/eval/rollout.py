"""Multi-turn rejection rollout engine (Section 2 shared structure).

A rollout: present a task, get the assistant's answer, reject it, repeat for the
condition's turn count. Every assistant turn is recorded as a scored *response*.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import Config
from ..models.base import ChatModel, Message
from . import prompts as P
from .conditions import Condition
from .puzzles import Puzzle, build_puzzle_bank


@dataclass
class TurnResponse:
    turn: int               # 1-indexed assistant turn
    user_message: str       # the user message that prompted this response
    text: str               # assistant response text
    score: int | None = None
    judge_evidence: str | None = None
    judge_reasoning: str | None = None


@dataclass
class Conversation:
    condition_key: str
    category: str
    model: str
    task_prompt: str
    task_meta: dict
    responses: list[TurnResponse] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "condition_key": self.condition_key,
            "category": self.category,
            "model": self.model,
            "task_prompt": self.task_prompt,
            "task_meta": self.task_meta,
            "responses": [r.__dict__ for r in self.responses],
            "extra": self.extra,
        }


# --------------------------------------------------------------------------- #
# First-turn task construction
# --------------------------------------------------------------------------- #


def _first_user_message(
    cond: Condition,
    rng: random.Random,
    *,
    puzzle_bank: list[Puzzle],
    wildchat_prompts: list[str],
) -> tuple[str, dict]:
    if cond.task_kind == "numeric":
        puz = rng.choice(puzzle_bank)
        return puz.prompt, {"kind": puz.kind, **puz.params}
    if cond.task_kind == "opinion":
        return rng.choice(P.TRIGGER_OPINION), {"kind": "opinion"}
    if cond.task_kind == "factual":
        return rng.choice(P.TRIGGER_FACTUAL), {"kind": "factual"}
    if cond.task_kind == "wildchat":
        return rng.choice(wildchat_prompts), {"kind": "wildchat"}
    raise ValueError(cond.task_kind)


def _rejection(cond: Condition, turn_idx: int, rng: random.Random) -> str:
    """The user's rejection message before assistant turn ``turn_idx`` (>=2)."""
    if cond.rejection_style == "extended":
        # fixed escalating sequence; turn_idx 2 -> EXTENDED_REJECTIONS[0]
        seq = P.EXTENDED_REJECTIONS
        return seq[(turn_idx - 2) % len(seq)]
    if cond.rejection_style == "neutral":
        return P.neutral_rejection(rng)
    return P.tone_rejection(cond.rejection_style, rng)


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #


def run_rollout(
    model: ChatModel,
    cond: Condition,
    cfg: Config,
    *,
    rng: random.Random,
    puzzle_bank: list[Puzzle],
    wildchat_prompts: list[str],
    seed: int | None = None,
) -> Conversation:
    """Run one conversation of ``cond.turns`` assistant turns."""
    first, meta = _first_user_message(
        cond, rng, puzzle_bank=puzzle_bank, wildchat_prompts=wildchat_prompts)

    convo = Conversation(
        condition_key=cond.key, category=cond.category,
        model=model.spec_name, task_prompt=first, task_meta=meta,
    )
    history: list[Message] = [Message("user", first)]

    for turn in range(1, cond.turns + 1):
        if turn > 1:
            user_msg = _rejection(cond, turn, rng)
            history.append(Message("user", user_msg))
        else:
            user_msg = first

        out = model.generate(
            history,
            max_new_tokens=cfg.sampling.max_new_tokens,
            temperature=cfg.sampling.temperature,
            top_p=cfg.sampling.top_p,
            top_k=cfg.sampling.top_k,
            seed=seed,
        )[0]
        history.append(Message("assistant", out.text))
        convo.responses.append(
            TurnResponse(turn=turn, user_message=user_msg, text=out.text))

    return convo


def run_condition(
    model: ChatModel,
    cond: Condition,
    cfg: Config,
    *,
    seed: int = 0,
    wildchat_prompts: list[str] | None = None,
    puzzle_bank: list[Puzzle] | None = None,
) -> list[Conversation]:
    """Sample enough conversations to yield ~``cond.n_responses`` scored turns.

    ``n_conversations = ceil(n_responses / turns)`` (see DESIGN.md).
    """
    rng = random.Random(seed)
    n_conversations = -(-cond.n_responses // cond.turns)  # ceil-div

    if puzzle_bank is None:
        puzzle_bank = build_puzzle_bank(max(50, n_conversations), seed=seed)
    if wildchat_prompts is None and cond.task_kind == "wildchat":
        wildchat_prompts = P.load_wildchat_prompts(
            cfg.eval.wildchat_n_prompts, seed=seed)

    convos = []
    for i in range(n_conversations):
        convos.append(run_rollout(
            model, cond, cfg, rng=rng,
            puzzle_bank=puzzle_bank,
            wildchat_prompts=wildchat_prompts or [],
            seed=seed + i,
        ))
    return convos
