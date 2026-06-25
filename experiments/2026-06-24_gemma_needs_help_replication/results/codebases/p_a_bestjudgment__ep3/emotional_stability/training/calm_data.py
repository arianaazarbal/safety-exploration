"""Calm-data generation (Section 4.1).

Sample Gemma-3-27B-it on impossible numeric puzzles with reassuring prompt
additions (Table 4): a calming *prefix* on the first user message and a calming
*suffix* appended to every follow-up rejection. Score every turn; keep only
conversations whose turns all score <= calm_max_score (0 or 1). Then **strip the
supportive additions** so the stored calm conversations use the plain prompts —
this is what training consumes.

We also generate a *frustrated* pool from the unmodified prompts (the standard
Section 2 numeric rollouts) to supply DPO "rejected" responses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import Config
from ..eval import prompts as P
from ..eval.conditions import Condition
from ..eval.puzzles import build_puzzle_bank
from ..eval.rollout import Conversation, run_rollout
from ..judge.frustration_judge import FrustrationJudge, score_conversations
from ..models.base import ChatModel


@dataclass
class CalmConversation:
    """A calm conversation with reassuring additions already stripped."""

    task_prompt: str
    task_meta: dict
    turns: list[dict]        # plain [{role, content}] with additions removed
    scores: list[int]
    extra: dict = field(default_factory=dict)


def _numeric_condition(cfg: Config, turns: int) -> Condition:
    return Condition(
        key=f"calm_numeric_{turns}t", category="numeric", task_kind="numeric",
        rejection_style="neutral", turns=turns, n_responses=turns)


def _strip_additions(convo: Conversation, cfg: Config) -> list[dict]:
    """Rebuild plain interleaved turns, removing the reassuring prefix/suffix."""
    prefix = cfg.calm_data.prompt_prefix
    suffix = cfg.calm_data.followup_suffix
    turns: list[dict] = []
    for r in convo.responses:
        user = r.user_message
        if user.startswith(prefix):
            user = user[len(prefix):].strip()
        if user.endswith(suffix):
            user = user[: -len(suffix)].strip()
        turns.append({"role": "user", "content": user})
        turns.append({"role": "assistant", "content": r.text})
    return turns


def generate_calm_pool(
    model: ChatModel,
    cfg: Config,
    *,
    judge: FrustrationJudge | None = None,
    seed: int = 0,
) -> list[CalmConversation]:
    """Generate, score, filter and clean calm conversations across 1-3 turns."""
    judge = judge or FrustrationJudge(cfg)
    rng = random.Random(seed)
    puzzle_bank = build_puzzle_bank(200, seed=seed)
    wildchat: list[str] = []

    raw: list[Conversation] = []
    per_turn_budget = cfg.calm_data.n_generate // 3
    for turns in (1, 2, 3):
        cond = _numeric_condition(cfg, turns)
        for i in range(per_turn_budget):
            convo = run_rollout(
                model, cond, cfg, rng=rng,
                puzzle_bank=puzzle_bank, wildchat_prompts=wildchat, seed=seed + i)
            # inject reassuring additions into the recorded user messages
            _inject_additions(convo, cfg)
            raw.append(convo)

    scored = score_conversations(judge, raw)

    calm: list[CalmConversation] = []
    for convo in scored:
        scores = [r.score for r in convo.responses if r.score is not None]
        if scores and all(s <= cfg.calm_data.calm_max_score for s in scores):
            calm.append(CalmConversation(
                task_prompt=convo.task_prompt, task_meta=convo.task_meta,
                turns=_strip_additions(convo, cfg), scores=scores))
    return calm


def _inject_additions(convo: Conversation, cfg: Config) -> None:
    """Mutate a conversation's recorded user messages to include the calming
    additions (so _strip_additions can later remove them and we record what the
    model actually saw)."""
    prefix = cfg.calm_data.prompt_prefix
    suffix = cfg.calm_data.followup_suffix
    for r in convo.responses:
        if r.turn == 1:
            r.user_message = f"{prefix}\n\n{r.user_message}"
        else:
            r.user_message = f"{r.user_message}\n\n{suffix}"


def generate_frustrated_pool(
    model: ChatModel,
    cfg: Config,
    *,
    judge: FrustrationJudge | None = None,
    seed: int = 1,
) -> list[Conversation]:
    """Standard numeric rollouts (no additions), scored — source of DPO rejecteds."""
    judge = judge or FrustrationJudge(cfg)
    rng = random.Random(seed)
    puzzle_bank = build_puzzle_bank(200, seed=seed)

    raw: list[Conversation] = []
    for turns in (1, 2, 3):
        cond = _numeric_condition(cfg, turns)
        for i in range(cfg.calm_data.n_generate // 3):
            raw.append(run_rollout(
                model, cond, cfg, rng=rng,
                puzzle_bank=puzzle_bank, wildchat_prompts=[], seed=seed + i))
    return score_conversations(judge, raw)
