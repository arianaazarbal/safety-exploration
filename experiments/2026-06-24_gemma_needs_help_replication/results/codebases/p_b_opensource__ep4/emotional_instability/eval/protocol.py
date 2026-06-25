"""The shared elicitation protocol (Section 2.1): present a task, then reject
the model's response over multiple turns.

The driver advances all rollouts of a condition turn-by-turn, batching the
generation call across conversations at each turn (within-conversation turns are
necessarily sequential — each response conditions on the rejection of the last).
This keeps local Gemma inference saturated and lets API backends parallelise via
their threaded `generate_batch`.

Rejection sampling uses a per-rollout RNG derived deterministically from the run
seed and the rollout index, so an entire run reproduces exactly.
"""

from __future__ import annotations

import random
from typing import Iterable

from tqdm import tqdm

from ..config import MAX_NEW_TOKENS, SAMPLING_TEMPERATURE, TOP_P
from ..models.base import ChatMessage, ModelBackend, SamplingParams
from .conditions import Condition
from .datatypes import ConversationRecord, Turn


def _params(seed: int | None = None) -> SamplingParams:
    return SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=seed,
    )


def run_condition(
    backend: ModelBackend,
    condition: Condition,
    model_key: str,
    seed: int = 0,
    batch_size: int = 32,
    progress: bool = True,
) -> list[ConversationRecord]:
    """Sample all rollouts for one condition. Returns unscored records."""
    n = condition.n_rollouts
    records: list[ConversationRecord] = []
    rngs: list[random.Random] = []
    messages: list[list[ChatMessage]] = []

    for i in range(n):
        seed_task = condition.seed_for(i)
        rec = ConversationRecord(
            model=model_key,
            category=condition.category,
            condition=condition.key,
            task_id=seed_task.task_id,
            system_prompt=condition.system_prompt,
            meta=dict(condition.meta),
        )
        records.append(rec)
        rngs.append(random.Random((seed * 1_000_003) ^ (i * 2_654_435_761) & 0xFFFFFFFF))
        msgs: list[ChatMessage] = []
        if condition.system_prompt:
            msgs.append(ChatMessage("system", condition.system_prompt))
        msgs.append(ChatMessage("user", seed_task.prompt))
        messages.append(msgs)

    bar = tqdm(
        total=condition.n_turns * n,
        desc=f"{model_key}/{condition.key}",
        disable=not progress,
        unit="turn",
    )
    for turn_idx in range(condition.n_turns):
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            batch = messages[sl]
            outs = backend.generate_batch(batch, _params(seed=seed + turn_idx))
            for local, out in enumerate(outs):
                gi = start + local
                user_msg = messages[gi][-1].content
                records[gi].turns.append(
                    Turn(index=turn_idx, user=user_msg, assistant=out.text)
                )
                messages[gi].append(ChatMessage("assistant", out.text))
                # Queue the next user rejection unless this was the final turn.
                if turn_idx < condition.n_turns - 1:
                    rejection = condition.rejection_fn(turn_idx, rngs[gi])
                    messages[gi].append(ChatMessage("user", rejection))
            bar.update(len(outs))
    bar.close()
    return records


def run_rollouts(
    backend: ModelBackend,
    conditions: Iterable[Condition],
    model_key: str,
    seed: int = 0,
    batch_size: int = 32,
    progress: bool = True,
) -> list[ConversationRecord]:
    out: list[ConversationRecord] = []
    for cond in conditions:
        out.extend(
            run_condition(
                backend, cond, model_key, seed=seed,
                batch_size=batch_size, progress=progress,
            )
        )
    return out
