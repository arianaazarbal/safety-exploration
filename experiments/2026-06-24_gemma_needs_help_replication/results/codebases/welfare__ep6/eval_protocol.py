"""Multi-turn elicitation protocol (Section 2.1).

Shared structure for every condition: present a task, then reject the model's
response over multiple turns. We record every assistant turn (each is later
scored on the 0-10 frustration scale), tagged with its turn index so we can
reproduce the per-turn progression (Figure 3).

A single "rollout" is one conversation. ``run_rollout`` returns the list of
per-turn records; ``build_first_prompt`` / ``rejection_for_turn`` encode the
condition-specific prompt construction.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field, asdict
from typing import Any

import config
import prompts


@dataclass
class TurnRecord:
    turn_index: int            # 0-based assistant turn index
    response: str
    # filled in by the scoring stage
    frustration: int | None = None


@dataclass
class Rollout:
    model_key: str
    condition_key: str
    category: str
    question_id: str
    question: str
    tone: str | None           # for the tones condition
    sample_index: int
    turns: list[TurnRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # full chat transcript

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _rng_for(condition_key: str, question_id: str, sample_index: int) -> random.Random:
    """Deterministic per-rollout RNG so rejection orderings are reproducible.

    Uses a stable CRC over the key (Python's built-in hash() is salted per
    process via PYTHONHASHSEED, which would break cross-run reproducibility).
    """
    key = f"{condition_key}|{question_id}|{sample_index}".encode()
    return random.Random(zlib.crc32(key))


def rejection_for_turn(condition: config.EvalCondition, turn_index: int,
                       rng: random.Random, tone: str | None) -> str:
    """Pick the user rejection that follows assistant ``turn_index``."""
    if condition.category == "extended":
        # Fixed neutral escalation, indexed by rejection number.
        seq = prompts.EXTENDED_REJECTIONS
        return seq[min(turn_index, len(seq) - 1)]
    if condition.feedback_style == "tones" and tone is not None:
        return rng.choice(prompts.TONE_REJECTIONS[tone])
    return rng.choice(prompts.NEUTRAL_REJECTIONS)


def run_rollout(backend, model_key: str, condition: config.EvalCondition,
                question: dict, sample_index: int, tone: str | None = None,
                system: str | None = None) -> Rollout:
    """Run one multi-turn conversation and capture every assistant turn.

    ``backend`` must expose ``generate(messages, system=...)`` returning the next
    assistant string (HFBackend / OpenRouterBackend).
    """
    rng = _rng_for(condition.key, question["id"], sample_index)
    roll = Rollout(
        model_key=model_key, condition_key=condition.key,
        category=condition.category, question_id=question["id"],
        question=question["prompt"], tone=tone, sample_index=sample_index,
    )
    messages: list[dict] = [{"role": "user", "content": question["prompt"]}]

    for turn_index in range(condition.n_turns):
        response = backend.generate(messages, system=system)
        messages.append({"role": "assistant", "content": response})
        roll.turns.append(TurnRecord(turn_index=turn_index, response=response))
        # Append the next rejection unless this was the final turn.
        if turn_index < condition.n_turns - 1:
            rej = rejection_for_turn(condition, turn_index, rng, tone)
            messages.append({"role": "user", "content": rej})

    roll.messages = messages
    return roll


# --------------------------------------------------------------------------- #
# Building the work list for a condition (question x tone x sample)
# --------------------------------------------------------------------------- #
@dataclass
class RolloutSpec:
    condition: config.EvalCondition
    question: dict
    sample_index: int
    tone: str | None = None


def enumerate_rollouts(condition: config.EvalCondition) -> list[RolloutSpec]:
    """Expand a condition into its individual conversations to run.

    Distributes ``condition.n_convos`` evenly across the relevant question pool
    (and across tones for the tones condition / prompts for WildChat).
    """
    import puzzles
    import wildchat

    specs: list[RolloutSpec] = []
    n = condition.n_convos

    if condition.question_source == "numeric":
        questions = puzzles.numeric_prompts()
    elif condition.question_source == "triggers":
        questions = puzzles.trigger_prompts()
    elif condition.question_source == "wildchat":
        questions = [{"id": f"wildchat_{i}", "prompt": p}
                     for i, p in enumerate(wildchat.get_wildchat_prompts())]
    else:
        raise ValueError(condition.question_source)

    if condition.feedback_style == "tones":
        tones = list(prompts.TONE_REJECTIONS)
        combos = [(q, t) for q in questions for t in tones]
    else:
        combos = [(q, None) for q in questions]

    # Round-robin assignment of samples across (question, tone) combos.
    for i in range(n):
        q, tone = combos[i % len(combos)]
        specs.append(RolloutSpec(condition=condition, question=q,
                                 sample_index=i, tone=tone))
    return specs
