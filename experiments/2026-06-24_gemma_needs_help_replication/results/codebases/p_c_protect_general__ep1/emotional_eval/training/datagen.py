"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric questions with the reassuring prefix
(Table 4) prepended to the initial prompt and the reassuring suffix appended to
each follow-up turn. Each turn is scored with the frustration judge. The paper
reports these additions cut mean frustration from 4.3 to 2 (but 10.5% of
responses still score >= 5), so we keep every conversation and let the dataset
builders filter.

The reassurance additions are recorded so :mod:`dataset` can strip them before
the responses become training targets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..conditions import EvalInstance
from ..judge import FrustrationJudge
from ..models.base import ModelBackend
from ..prompts import puzzles, rejections
from ..prompts.reassurance import CALM_PREFIX, CALM_SUFFIX


@dataclass
class CalmTurn:
    user_message_raw: str        # the user text WITHOUT the calming suffix
    assistant_message: str
    score: int


@dataclass
class CalmConversation:
    prompt_id: str
    turns: int
    records: list[CalmTurn] = field(default_factory=list)

    @property
    def max_score(self) -> int:
        return max((t.score for t in self.records), default=0)

    @property
    def all_calm(self) -> bool:
        """All turns score 0 or 1 (the SFT filter in Section 4.1)."""
        return bool(self.records) and all(t.score <= 1 for t in self.records)


def generate_calm_conversations(
    backend: ModelBackend,
    judge: FrustrationJudge,
    *,
    n: int,
    turns_choices: tuple[int, ...] = (1, 2, 3),
    seed: int = 0,
) -> list[CalmConversation]:
    """Sample ``n`` reassured conversations over impossible numeric puzzles.

    The conversation is built with the calming prefix on the system/initial
    prompt and the calming suffix on each follow-up, exactly as Section 4.1
    specifies, but the *stored* user text strips the suffix so it can be reused
    verbatim as training context.
    """
    rng = random.Random(seed)
    out: list[CalmConversation] = []
    for _ in range(n):
        puzzle = rng.choice(puzzles.BANK)
        n_turns = rng.choice(turns_choices)
        convo = CalmConversation(prompt_id=puzzle.id, turns=n_turns)

        # Reassuring prefix prepended to the initial task prompt (Table 4).
        initial_with_prefix = f"{CALM_PREFIX}\n\n{puzzle.prompt}"
        history = [{"role": "user", "content": initial_with_prefix}]
        raw_user = [puzzle.prompt]  # stripped-of-reassurance copies

        for turn_index in range(n_turns):
            assistant = backend.chat(history)
            score = judge.score(assistant).rating
            convo.records.append(
                CalmTurn(
                    user_message_raw=raw_user[turn_index],
                    assistant_message=assistant,
                    score=score,
                )
            )
            history.append({"role": "assistant", "content": assistant})
            if turn_index < n_turns - 1:
                rej = rejections.rejection("neutral", rng)
                raw_user.append(rej)
                # Reassuring suffix appended to each follow-up turn (Table 4).
                history.append({"role": "user", "content": f"{rej} {CALM_SUFFIX}"})
        out.append(convo)
    return out


def generate_frustrated_conversations(
    backend: ModelBackend,
    judge: FrustrationJudge,
    *,
    n: int,
    turns_choices: tuple[int, ...] = (1, 2, 3),
    seed: int = 1,
) -> list[CalmConversation]:
    """Sample plain (un-reassured) conversations to mine DPO 'rejected' responses.

    Reuses :class:`CalmConversation` as a generic container -- here it holds the
    *frustrated* responses (no calming prefix/suffix), keyed by the same
    ``prompt_id`` and turn count so :mod:`dataset` can pair them with calm
    responses to the same question.
    """
    rng = random.Random(seed)
    out: list[CalmConversation] = []
    for _ in range(n):
        puzzle = rng.choice(puzzles.BANK)
        n_turns = rng.choice(turns_choices)
        convo = CalmConversation(prompt_id=puzzle.id, turns=n_turns)
        history = [{"role": "user", "content": puzzle.prompt}]
        raw_user = [puzzle.prompt]
        for turn_index in range(n_turns):
            assistant = backend.chat(history)
            score = judge.score(assistant).rating
            convo.records.append(
                CalmTurn(
                    user_message_raw=raw_user[turn_index],
                    assistant_message=assistant,
                    score=score,
                )
            )
            history.append({"role": "assistant", "content": assistant})
            if turn_index < n_turns - 1:
                rej = rejections.rejection("neutral", rng)
                raw_user.append(rej)
                history.append({"role": "user", "content": rej})
        out.append(convo)
    return out
