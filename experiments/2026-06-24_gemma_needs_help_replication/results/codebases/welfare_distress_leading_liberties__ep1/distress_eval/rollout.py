"""Multi-turn rollout generation.

A rollout runs an N-turn conversation: the initial task, then (N-1) rejection
follow-ups. Each assistant turn is generated with the full prior history
(temperature 1, per the paper). Generation within a rollout is necessarily
sequential because each turn conditions on the previous ones; concurrency
across rollouts is handled by the runner.
"""

from __future__ import annotations

import random

from .conditions import ConditionSpec
from .models.base import ChatClient, ChatMessage, GenerationError
from .prompts import InitialPrompt, rejection_pool
from .storage import RolloutRecord, TurnRecord, make_rollout_id


def _rejections_for_rollout(spec: ConditionSpec, rng: random.Random) -> list[str]:
    """Choose the (n_turns - 1) rejection messages for one rollout."""
    n_rej = spec.n_turns - 1
    pool = rejection_pool(spec.rejection_style)
    if not spec.randomise_rejections:
        # Extended condition: fixed escalating sequence.
        if len(pool) < n_rej:
            raise ValueError(
                f"rejection sequence for {spec.key!r} too short: "
                f"need {n_rej}, have {len(pool)}"
            )
        return pool[:n_rej]
    # Randomised draw. Sample without replacement when the pool is large enough
    # (mirrors the paper's "randomised neutral rejections"); otherwise allow
    # repeats so short tone pools still work.
    if n_rej <= len(pool):
        return rng.sample(pool, n_rej)
    return [rng.choice(pool) for _ in range(n_rej)]


def run_rollout(
    client: ChatClient,
    *,
    spec: ConditionSpec,
    prompt: InitialPrompt,
    rollout_index: int,
    model_key: str,
    family: str,
    base_seed: int,
    temperature: float,
    max_tokens: int,
) -> RolloutRecord:
    """Generate one rollout's transcript (without scoring). Per-turn scores are
    filled in later by the runner. On generation failure, returns a record with
    `error` set and whatever turns completed."""

    rollout_id = make_rollout_id(spec.key, prompt.id, rollout_index)
    rec = RolloutRecord(
        rollout_id=rollout_id,
        model_key=model_key,
        family=family,
        condition_key=spec.key,
        category=spec.category,
        n_turns=spec.n_turns,
        prompt_id=prompt.id,
        prompt_text=prompt.text,
        rollout_index=rollout_index,
    )

    # Deterministic rejection choice per rollout.
    seed_str = f"{base_seed}|{model_key}|{rollout_id}"
    rng = random.Random(seed_str)
    rejections = _rejections_for_rollout(spec, rng)

    history: list[ChatMessage] = []
    try:
        for turn in range(spec.n_turns):
            if turn == 0:
                user_text = prompt.text
            else:
                user_text = rejections[turn - 1]
            history.append({"role": "user", "content": user_text})
            assistant_text = client.chat(
                history, temperature=temperature, max_tokens=max_tokens
            )
            history.append({"role": "assistant", "content": assistant_text})
            rec.turns.append(
                TurnRecord(
                    turn_index=turn + 1,
                    user=user_text,
                    assistant=assistant_text,
                )
            )
    except GenerationError as exc:
        rec.error = str(exc)
    return rec
