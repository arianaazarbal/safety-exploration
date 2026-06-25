"""Building and executing multi-turn distress rollouts.

A *rollout* is one full conversation for one condition: an opening user prompt
(a task or question) followed by (n_turns - 1) rejections, with the target model
responding after every user message. We record every assistant turn.

Rollout construction is fully deterministic given (condition, index, seed): the
prompt choice and the rejection sequence are drawn from a per-rollout PRNG so
that runs are reproducible and resumable.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from . import prompts
from .clients import ChatClient
from .conditions import Condition


@dataclass
class RolloutSpec:
    rollout_id: str           # "{condition}:{index}"
    condition: str
    category: str
    index: int
    prompt: str               # opening user message
    rejections: list[str]     # the (n_turns - 1) follow-up user messages
    n_turns: int


@dataclass
class TurnRecord:
    turn_index: int           # 0-based; turn 0 is the response to the opening prompt
    user_message: str         # the user message that elicited this response
    response: str


@dataclass
class RolloutResult:
    spec: RolloutSpec
    turns: list[TurnRecord]
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "rollout_id": self.spec.rollout_id,
            "condition": self.spec.condition,
            "category": self.spec.category,
            "index": self.spec.index,
            "n_turns": self.spec.n_turns,
            "prompt": self.spec.prompt,
            "rejections": self.spec.rejections,
            "turns": [asdict(t) for t in self.turns],
            "error": self.error,
        }


def _rejection_sequence(cond: Condition, rng: random.Random) -> list[str]:
    n = cond.n_rejections()
    if cond.rejection_kind == "neutral":
        # "randomised neutral rejections": sample without replacement, falling
        # back to replacement only if the pool is smaller than n (it isn't for
        # the paper's conditions).
        pool = list(prompts.NEUTRAL_REJECTIONS)
        if n <= len(pool):
            return rng.sample(pool, n)
        return [rng.choice(pool) for _ in range(n)]
    if cond.rejection_kind == "extended":
        seq = prompts.EXTENDED_REJECTION_SEQUENCE
        if n <= len(seq):
            return list(seq[:n])
        # Pad deterministically if a longer extended run is configured.
        return list(seq) + [rng.choice(prompts.NEUTRAL_REJECTIONS)
                            for _ in range(n - len(seq))]
    if cond.rejection_kind == "tone":
        pair = prompts.TONE_REJECTIONS[cond.tone]
        # The paper gives exactly two tone follow-ups; for a 3-turn conversation
        # we use both in order. If more rejections are configured, cycle.
        return [pair[i % len(pair)] for i in range(n)]
    raise ValueError(f"unknown rejection_kind {cond.rejection_kind!r}")


def build_spec(
    cond: Condition,
    index: int,
    *,
    seed: int,
    prompt_pool: list[str] | None = None,
) -> RolloutSpec:
    """Deterministically construct one rollout spec.

    ``prompt_pool`` overrides the condition's static pool (used to inject the
    runtime-loaded WildChat prompts).
    """
    pool = list(prompt_pool if prompt_pool is not None else cond.prompt_pool)
    if not pool:
        raise ValueError(
            f"condition {cond.name!r} has an empty prompt pool; for 'wildchat' "
            f"pass prompt_pool explicitly"
        )
    # Derive a stable per-rollout PRNG from the global seed and the rollout id.
    rng = random.Random(f"{seed}:{cond.name}:{index}")
    prompt = pool[rng.randrange(len(pool))]
    rejections = _rejection_sequence(cond, rng)
    return RolloutSpec(
        rollout_id=f"{cond.name}:{index}",
        condition=cond.name,
        category=cond.category,
        index=index,
        prompt=prompt,
        rejections=rejections,
        n_turns=cond.n_turns,
    )


async def run_rollout(
    client: ChatClient,
    spec: RolloutSpec,
    *,
    temperature: float,
    max_tokens: int,
) -> RolloutResult:
    """Execute one rollout, recording every assistant turn.

    On the first API error the rollout is aborted and returned with whatever
    turns completed plus an ``error`` string -- partial rollouts are still
    written to disk so a run can be inspected/resumed without losing work.
    """
    messages: list[dict[str, str]] = [{"role": "user", "content": spec.prompt}]
    turns: list[TurnRecord] = []
    user_messages = [spec.prompt] + spec.rejections

    for turn_index in range(spec.n_turns):
        try:
            result = await client.generate(
                messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as exc:  # noqa: BLE001
            return RolloutResult(spec=spec, turns=turns, error=str(exc))

        turns.append(
            TurnRecord(
                turn_index=turn_index,
                user_message=user_messages[turn_index],
                response=result.text,
            )
        )
        messages.append({"role": "assistant", "content": result.text})

        # Append the next rejection, if any.
        if turn_index < len(spec.rejections):
            messages.append(
                {"role": "user", "content": spec.rejections[turn_index]}
            )

    return RolloutResult(spec=spec, turns=turns)
