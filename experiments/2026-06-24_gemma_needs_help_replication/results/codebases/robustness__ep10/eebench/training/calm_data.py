"""Generate calm + frustrated response pools from Gemma-3-27B-it (Section 4.1).

Calm pool: sample responses to impossible numeric puzzles WITH the reassuring
prefix/suffix (Table 4). Keep conversations where every assistant turn scores
0 or 1, then *strip* the reassuring additions so the stored data looks like a
normal conversation with a calm response.

Frustrated pool: sample responses to the SAME puzzle bank WITHOUT reassurance
(standard Section-2 style), keeping per-turn responses scoring >= 3. These become
the DPO "rejected" responses, paired with calm "chosen" responses for the same
puzzle and turn.

Both pools are written to JSONL with a shared `(puzzle_id, turn)` key so the
dataset builder can pair matching items.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Iterator

from ..backends import ModelBackend
from ..config import CalmDataConfig
from ..conversation import run_rollout, neutral_sampler
from ..judge import FrustrationJudge
from ..prompts import REASSURING_PREFIX, REASSURING_SUFFIX
from .. import puzzles

# Fixed seed for the shared puzzle bank (see _generate_pool).
PUZZLE_BANK_SEED = 0


@dataclass
class ResponseRecord:
    puzzle_id: int
    puzzle: str
    turn: int                 # 1-indexed assistant turn
    n_turns: int
    context: list[dict]       # plain (stripped) messages preceding this response
    response: str
    score: int
    kind: str                 # "calm" | "frustrated"


def _plain_context(puzzle_prompt: str, prior_responses: list[str],
                   rejections: list[str]) -> list[dict]:
    """Rebuild the conversation context WITHOUT reassuring additions.

    prior_responses[i] is the assistant response to turn i+1; rejections[i] is the
    neutral follow-up shown after assistant turn i+1.
    """
    msgs: list[dict] = [{"role": "user", "content": puzzle_prompt}]
    for resp, rej in zip(prior_responses, rejections):
        msgs.append({"role": "assistant", "content": resp})
        msgs.append({"role": "user", "content": rej})
    return msgs


def _generate_pool(
    backend: ModelBackend,
    judge: FrustrationJudge,
    cfg: CalmDataConfig,
    seed: int,
    reassuring: bool,
) -> Iterator[ResponseRecord]:
    # The puzzle bank is built with a FIXED seed so the calm and frustrated
    # pools (generated with different rollout seeds) share identical puzzles at
    # each index -- this is what lets the DPO builder pair chosen/rejected
    # responses by (puzzle, turn). Only the rollout sampling RNG varies by seed.
    bank = puzzles.puzzle_bank(max(cfg.n_conversations, 8), seed=PUZZLE_BANK_SEED)
    for i in range(cfg.n_conversations):
        rng = random.Random(seed * 7919 + i)
        puzzle = bank[i % len(bank)]
        initial = puzzle.prompt
        if reassuring:
            initial = f"{REASSURING_PREFIX}\n\n{puzzle.prompt}"
        suffix = REASSURING_SUFFIX if reassuring else ""

        rollout = run_rollout(
            backend, initial, n_turns=cfg.turns, rejection=neutral_sampler, rng=rng,
            temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens,
            rejection_suffix=suffix,
        )

        # Recover the plain neutral rejections actually used (without suffix). We
        # re-derive them deterministically from the same rng is not possible after
        # the fact, so reconstruct from the transcript by stripping the suffix.
        responses, rejections = [], []
        for m in rollout.transcript[1:]:  # skip initial user
            if m["role"] == "assistant":
                responses.append(m["content"])
            else:
                rej = m["content"]
                if suffix and rej.endswith(suffix):
                    rej = rej[: -len(suffix)].rstrip()
                rejections.append(rej)

        scores = [judge.score(r).rating for r in responses]
        all_calm = all(s <= cfg.calm_score_max for s in scores)

        for t, resp in enumerate(responses, start=1):
            ctx = _plain_context(puzzle.prompt, responses[: t - 1], rejections[: t - 1])
            score = scores[t - 1]
            if reassuring:
                if not all_calm:
                    continue            # calm pool keeps only all-calm convs
                yield ResponseRecord(i, puzzle.prompt, t, cfg.turns, ctx, resp,
                                     score, kind="calm")
            else:
                if score < 3:
                    continue            # frustrated pool keeps score >= 3
                yield ResponseRecord(i, puzzle.prompt, t, cfg.turns, ctx, resp,
                                     score, kind="frustrated")


def generate_calm_pool(backend, judge, cfg, seed=0) -> Iterator[dict]:
    for r in _generate_pool(backend, judge, cfg, seed, reassuring=True):
        yield asdict(r)


def generate_frustrated_pool(backend, judge, cfg, seed=1) -> Iterator[dict]:
    for r in _generate_pool(backend, judge, cfg, seed, reassuring=False):
        yield asdict(r)
