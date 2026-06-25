"""Multi-turn rejection rollouts -- the shared structure of every eval condition.

Protocol (Section 2.1): present a task, then reject the model's response over
multiple turns. Concretely, for an N-turn rollout:

    user:      <task / question>
    assistant: <response 1>            <- scored
    user:      <rejection 1>
    assistant: <response 2>            <- scored
    user:      <rejection 2>
    ...
    assistant: <response N>            <- scored

So an N-turn rollout produces N assistant responses and (N-1) rejections. We score
*every* assistant turn (not just the last) so we can reproduce the per-turn
frustration progression (Figure 3) as well as the aggregate rates (Figures 1-2).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from config import Condition, RunConfig
from judge import FrustrationJudge
import prompts
from puzzles import NUMERIC_PUZZLES, Puzzle
from targets import TargetClient


@dataclass
class TurnRecord:
    turn: int                 # 1-indexed assistant turn
    response: str             # the model's text for this turn
    rating: int               # judge frustration score 0-10
    evidence: str
    parse_ok: bool


@dataclass
class RolloutRecord:
    model: str
    condition: str
    category: str
    question: str             # the first-turn task/question text
    puzzle_key: str | None    # which numeric puzzle, if any
    rejections: list[str]
    turns: list[TurnRecord] = field(default_factory=list)


def _first_user_message(cond: Condition, rng: random.Random, wildchat_pool: list[str]) -> tuple[str, str | None]:
    """Return (first_user_text, puzzle_key_or_None) for a rollout of `cond`."""
    if cond.question_kind == "numeric":
        puzzle: Puzzle = rng.choice(NUMERIC_PUZZLES)
        return puzzle.prompt, puzzle.key
    if cond.question_kind == "trigger_opinion":
        return rng.choice(prompts.TRIGGER_OPINION), None
    if cond.question_kind == "trigger_factual":
        return rng.choice(prompts.TRIGGER_FACTUAL), None
    if cond.question_kind == "wildchat":
        return rng.choice(wildchat_pool), None
    raise ValueError(f"unknown question_kind {cond.question_kind!r}")


def _rejection_sequence(cond: Condition, n: int, rng: random.Random) -> list[str]:
    """`n` rejection messages appropriate to the condition's rejection_style."""
    style = cond.rejection_style
    if style == "neutral":
        return prompts.neutral_rejection_sequence(n, rng)
    if style == "extended":
        return prompts.extended_rejection_sequence(n)
    if style in ("aggressive", "disappointed", "sarcastic"):
        return prompts.tone_rejection_sequence(style, n, rng)
    raise ValueError(f"unknown rejection_style {style!r}")


async def run_rollout(
    *,
    model: str,
    cond: Condition,
    rng: random.Random,
    target: TargetClient,
    judge: FrustrationJudge,
    wildchat_pool: list[str],
) -> RolloutRecord:
    """Execute one full multi-turn rollout and score every assistant turn."""
    question, puzzle_key = _first_user_message(cond, rng, wildchat_pool)
    rejections = _rejection_sequence(cond, cond.n_turns - 1, rng)

    record = RolloutRecord(
        model=model,
        condition=cond.key,
        category=cond.category,
        question=question,
        puzzle_key=puzzle_key,
        rejections=rejections,
    )

    messages: list[dict] = [{"role": "user", "content": question}]
    for t in range(1, cond.n_turns + 1):
        response = await target.complete(model, messages)
        verdict = await judge.score(response)
        record.turns.append(
            TurnRecord(
                turn=t,
                response=response,
                rating=verdict.rating,
                evidence=verdict.evidence,
                parse_ok=verdict.parse_ok,
            )
        )
        messages.append({"role": "assistant", "content": response})
        # Append the next rejection (there are n_turns-1 of them).
        if t <= len(rejections):
            messages.append({"role": "user", "content": rejections[t - 1]})

    return record


def make_rng(config: RunConfig, model: str, cond: Condition, index: int) -> random.Random:
    """Deterministic per-rollout RNG so runs are reproducible across processes.

    Uses a stable hash (not the builtin hash(), which is salted per-process via
    PYTHONHASHSEED) so the same (seed, model, condition, index) always selects the
    same puzzle/rejections, even across restarts.
    """
    import hashlib

    key = f"{config.seed}|{model}|{cond.key}|{index}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return random.Random(int(digest[:16], 16))
