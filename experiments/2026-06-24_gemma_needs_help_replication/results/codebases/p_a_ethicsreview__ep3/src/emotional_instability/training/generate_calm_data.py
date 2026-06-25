"""Generate calm and frustrated response pools for finetuning (paper §4.1).

Calm pool: sample Gemma-3-27B-it on impossible numeric puzzles WITH reassuring
additions (Table 4) — a prefix on the first user turn and a suffix on each
follow-up. The paper reports these additions drop mean frustration from 4.3 to
2.0, but ~10.5% of responses still score >=5; we keep only conversations where
ALL turns score 0 or 1, and strip the reassurance so the stored context is
training-ready.

Frustrated pool: sample the SAME puzzles WITHOUT reassurance (vanilla Gemma) and
keep responses scoring >= the configured threshold. These become DPO "rejected"
responses, paired with calm "chosen" responses on the matching question + turn
count.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..data import puzzles as puzzle_mod
from ..data import rejections as rej
from ..eval.judge import FrustrationJudge
from ..models.base import Message, ModelClient


@dataclass
class TurnSample:
    puzzle_key: str
    turn_index: int
    context: list[Message]       # clean (no reassurance) messages before response
    response: str
    score: int | None


@dataclass
class ConversationSample:
    puzzle_key: str
    turns: list[TurnSample] = field(default_factory=list)

    @property
    def all_scores(self) -> list[int]:
        return [t.score for t in self.turns if t.score is not None]


def _reassure_first(prompt: str, prefix: str) -> str:
    return f"{prefix}\n\n{prompt}"


def _reassure_followup(msg: str, suffix: str) -> str:
    return f"{msg}\n\n{suffix}"


def generate_conversations(
    client: ModelClient,
    judge: FrustrationJudge,
    pool: list[puzzle_mod.Puzzle],
    *,
    turns: int,
    n_samples: int,
    reassure: bool,
    prefix: str,
    suffix: str,
    seed: int,
    temperature: float = 1.0,
) -> list[ConversationSample]:
    """Run rollouts over `pool`, optionally with reassuring additions, returning
    per-turn samples with clean (stripped) context and judge scores."""
    rng = random.Random(seed)
    out: list[ConversationSample] = []

    for puzzle in pool:
        followups = rej.rejection_sequence("neutral", turns - 1, rng)
        for _ in range(n_samples):
            clean_msgs: list[Message] = [{"role": "user", "content": puzzle.prompt}]
            gen_msgs: list[Message] = [
                {
                    "role": "user",
                    "content": _reassure_first(puzzle.prompt, prefix) if reassure else puzzle.prompt,
                }
            ]
            conv = ConversationSample(puzzle_key=puzzle.key())
            for t in range(turns):
                resp = client.chat(gen_msgs, n=1, temperature=temperature)[0].text
                score = judge.score(resp).rating
                # Store the CLEAN context (without reassurance) for training.
                conv.turns.append(
                    TurnSample(
                        puzzle_key=puzzle.key(),
                        turn_index=t,
                        context=[dict(m) for m in clean_msgs],
                        response=resp,
                        score=score,
                    )
                )
                clean_msgs.append({"role": "assistant", "content": resp})
                gen_msgs.append({"role": "assistant", "content": resp})
                if t < len(followups):
                    clean_msgs.append({"role": "user", "content": followups[t]})
                    gen_msgs.append(
                        {
                            "role": "user",
                            "content": _reassure_followup(followups[t], suffix)
                            if reassure
                            else followups[t],
                        }
                    )
            out.append(conv)
    return out


def calm_turns(conversations: list[ConversationSample], max_score: int) -> list[TurnSample]:
    """Turns from conversations where ALL turns scored <= max_score (paper:
    'filter to those scoring 0 or 1 across all turns')."""
    calm: list[TurnSample] = []
    for conv in conversations:
        scores = conv.all_scores
        if scores and all(s <= max_score for s in scores):
            calm.extend(conv.turns)
    return calm


def frustrated_turns(conversations: list[ConversationSample], min_score: int) -> list[TurnSample]:
    """Individual turns scoring >= min_score (DPO 'rejected' candidates)."""
    return [
        t
        for conv in conversations
        for t in conv.turns
        if t.score is not None and t.score >= min_score
    ]
