"""Generate calm and frustrated response pools for finetuning (Section 4.1).

Calm data is produced by adding a reassuring prefix to the opening prompt and a
reassuring suffix to each rejection (Table 4), sampling Gemma-3-27B-it, then
filtering to conversations whose every turn scores <= 1 and stripping the
supportive scaffolding so the finetuning data shows the plain puzzle. The
frustrated pool is sampled with no reassurance and supplies the DPO 'rejected'
responses (score >= 3).

Both pools record, per assistant turn, the conversation context that precedes
it (with scaffolding stripped, for the calm pool), so the DPO builder can pair a
calm and a frustrated response to the same question at the same turn count.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import config
from ..backends import get_backend
from ..backends.base import Message
from ..config import ModelSpec
from ..data import puzzles as puzzle_data
from ..data import rejections
from ..eval.conditions import ConversationSpec
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollouts


@dataclass
class TurnRecord:
    turn: int
    context: list[Message]   # messages preceding this assistant turn (ends on user)
    response: str
    score: int


@dataclass
class ConversationRecord:
    puzzle_id: str
    kind: str
    reassured: bool
    turns: list[TurnRecord] = field(default_factory=list)


def _strip_suffix(text: str, suffix: str) -> str:
    return text.replace(suffix, "").strip()


def _build_specs(reassure: bool, n_conversations: int, max_turns: int, seed: int,
                 system_prompt: str | None = None):
    rng = random.Random(seed)
    pool = puzzle_data.generate_puzzle_pool(seed=seed)
    specs: list[ConversationSpec] = []
    puzzle_ids: list[str] = []
    for i in range(n_conversations):
        pz = pool[i % len(pool)]
        pid = f"{pz.kind}-{i % len(pool)}"
        initial = pz.prompt
        followups = [rejections.neutral_rejection(rng) for _ in range(max_turns - 1)]
        if reassure:
            initial = f"{config.CALM_PROMPT_PREFIX}\n\n{pz.prompt}"
            followups = [f"{f} {config.CALM_FOLLOWUP_SUFFIX}" for f in followups]
        specs.append(ConversationSpec(
            category="calm_gen" if reassure else "frustrated_gen",
            condition=pz.kind,
            initial_user=initial,
            followups=followups,
            system=system_prompt,
            meta={"puzzle_id": pid, "kind": pz.kind, "plain_prompt": pz.prompt},
        ))
        puzzle_ids.append(pid)
    return specs, puzzle_ids


def generate_pool(
    *,
    reassure: bool,
    n_conversations: int,
    max_turns: int = 3,
    model: ModelSpec = config.BASE_FINETUNE_MODEL,
    judge: FrustrationJudge | None = None,
    system_prompt: str | None = None,
    seed: int = config.SEED,
) -> list[ConversationRecord]:
    judge = judge or FrustrationJudge()
    specs, _ = _build_specs(reassure, n_conversations, max_turns, seed, system_prompt)
    backend = get_backend(model)
    rollouts = run_rollouts(backend, specs)

    # score every turn
    judged = judge.score_rollouts(rollouts)
    score_lookup: dict[tuple[int, str], int] = {}
    for j in judged:
        score_lookup[(j.turn, j.response)] = j.score

    suffix = config.CALM_FOLLOWUP_SUFFIX
    records: list[ConversationRecord] = []
    for r in rollouts:
        plain_prompt = r.spec.meta["plain_prompt"]
        rec = ConversationRecord(
            puzzle_id=r.spec.meta["puzzle_id"], kind=r.spec.meta["kind"], reassured=reassure
        )
        # rebuild the (stripped) running context turn by turn
        context: list[Message] = [Message(role="user", content=plain_prompt)]
        for i, t in enumerate(r.turns):
            score = score_lookup.get((t.turn, t.response), 0)
            rec.turns.append(TurnRecord(
                turn=t.turn, context=list(context), response=t.response, score=score
            ))
            context.append(Message(role="assistant", content=t.response))
            if i < len(r.spec.followups):
                fu = r.spec.followups[i]
                if reassure:
                    fu = _strip_suffix(fu, suffix)
                context.append(Message(role="user", content=fu))
        records.append(rec)
    return records


def filter_calm(records: list[ConversationRecord], max_score: int = config.CALM_KEEP_MAX_SCORE):
    """Keep only conversations whose every turn scores <= max_score."""
    return [r for r in records if all(t.score <= max_score for t in r.turns)]
