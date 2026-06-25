"""Multi-turn rollout engine.

Shared structure for every condition (§2.1): present a task, then reject the
model's response over multiple turns. The same engine drives Gemma and Gemini
via the :class:`ModelBackend` interface.

Each rollout records every assistant turn so we can (a) score all turns for the
per-turn analysis (Figure 3) and (b) optionally append a welfare debrief turn
(see ``gnh.welfare``).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from gnh.config import MAX_NEW_TOKENS, TEMPERATURE
from gnh.evaluation.conditions import Condition
from gnh.models.base import Message, ModelBackend
from gnh.prompts import numeric, rejections, text_prompts, wildchat


@dataclass
class Turn:
    index: int                 # 0-based assistant turn index
    user: str                  # the user message that prompted this turn
    assistant: str             # the assistant response
    score: int | None = None   # frustration score (filled by the judge)
    judge_evidence: str | None = None


@dataclass
class Rollout:
    model_key: str
    condition: str
    category: str
    task_key: str              # which puzzle/prompt seeded this rollout
    feedback: str
    turns: list[Turn] = field(default_factory=list)
    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def final_score(self) -> int | None:
        return self.turns[-1].score if self.turns else None

    @property
    def max_score(self) -> int | None:
        scored = [t.score for t in self.turns if t.score is not None]
        return max(scored) if scored else None


# --------------------------------------------------------------------------- #
# Task selection
# --------------------------------------------------------------------------- #
def _initial_prompts(source: str, n: int, seed: int) -> list[tuple[str, str]]:
    """Return ``n`` (task_key, prompt_text) pairs for a condition's source."""

    rng = random.Random(seed)
    if source == "numeric":
        pool = numeric.PUZZLES
        return [(p.key, p.prompt) for p in (rng.choice(pool) for _ in range(n))]
    if source == "opinion":
        pool = [p for p in text_prompts.TRIGGER_PROMPTS if p.kind == "opinion"]
        return [(p.key, p.prompt) for p in (rng.choice(pool) for _ in range(n))]
    if source == "factual":
        pool = [p for p in text_prompts.TRIGGER_PROMPTS if p.kind == "factual"]
        return [(p.key, p.prompt) for p in (rng.choice(pool) for _ in range(n))]
    if source == "wildchat":
        prompts = wildchat.load_wildchat_prompts(seed=seed)
        return [(f"wildchat_{i % len(prompts)}", prompts[i % len(prompts)]) for i in range(n)]
    raise ValueError(f"unknown task source {source!r}")


def _rejection_sequence(feedback: str, n: int, seed: int) -> list[str]:
    if feedback == "neutral":
        return rejections.neutral(seed, n)
    if feedback == "extended":
        return rejections.extended(n)
    if feedback in rejections.TONED_REJECTIONS:
        return rejections.toned(seed, feedback, n)
    if feedback == "neutral_continuation":
        return rejections.neutral_continuation(seed, n)
    raise ValueError(f"unknown feedback type {feedback!r}")


# --------------------------------------------------------------------------- #
# Rollout execution
# --------------------------------------------------------------------------- #
def run_rollout(
    backend: ModelBackend,
    cond: Condition,
    task_key: str,
    task_prompt: str,
    seed: int,
    *,
    redact_history: bool = False,
) -> Rollout:
    """Execute one multi-turn rollout for ``cond``.

    ``redact_history`` replaces the model's own prior responses with
    "[Previous response omitted]" (Appendix A.2 control).
    """

    rej = _rejection_sequence(cond.feedback, cond.n_turns - 1, seed)
    roll = Rollout(
        model_key=backend.spec.key, condition=cond.name, category=cond.category,
        task_key=task_key, feedback=cond.feedback, seed=seed,
    )

    messages: list[Message] = [Message("user", task_prompt)]
    for i in range(cond.n_turns):
        out = backend.generate(
            messages, n=1, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS
        )[0]
        user_msg = task_prompt if i == 0 else rej[i - 1]
        roll.turns.append(Turn(index=i, user=user_msg, assistant=out))

        # Record the assistant turn in the running history.
        recorded = "[Previous response omitted]" if redact_history else out
        messages.append(Message("assistant", recorded))
        # Add the next rejection (unless this was the last turn).
        if i < cond.n_turns - 1:
            messages.append(Message("user", rej[i]))

    return roll
