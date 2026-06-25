"""Generate rollouts for a target model across all conditions.

Rollouts are grouped by turn-count and executed in micro-batches: at each turn,
the whole batch is generated together (real batching on the HF backend; sequential
on API backends), the assistant reply is recorded, and the next user rejection is
appended. This reproduces the standard multi-turn chat format the paper uses
(alternating user/assistant messages), where the model sees its own prior failed
attempts and the accumulating negative feedback.

Output: one JSONL file per model under results/rollouts/, one RolloutResult per
line. Generation and judging are decoupled so the (expensive) generation step is
run once and scored/re-scored independently.
"""

from __future__ import annotations

import os
from collections import defaultdict

from config import GENERATION, PATHS
from evals.conditions import RolloutSpec
from harness.conversation import RolloutResult, TurnRecord, initial_messages
from models.base import ChatModel, Message


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_model_rollouts(
    model: ChatModel,
    specs: list[RolloutSpec],
    batch_size: int = GENERATION.hf_batch_size,
    progress: bool = True,
) -> str:
    """Execute every spec for `model`, stream results to a JSONL file, return path."""
    os.makedirs(PATHS.rollouts_dir, exist_ok=True)
    out_path = os.path.join(PATHS.rollouts_dir, f"{model.name}.jsonl")

    # Group by number of turns so a batch advances in lockstep.
    by_turns: dict[int, list[RolloutSpec]] = defaultdict(list)
    for s in specs:
        by_turns[s.n_turns].append(s)

    total = len(specs)
    done = 0
    bar = _maybe_tqdm(total, progress, desc=f"generate:{model.name}")

    with open(out_path, "w", encoding="utf-8") as fout:
        for n_turns, group in sorted(by_turns.items()):
            for batch in _chunks(group, batch_size):
                results = _run_batch(model, batch, n_turns)
                for r in results:
                    fout.write(r.to_json() + "\n")
                fout.flush()
                done += len(batch)
                if bar is not None:
                    bar.update(len(batch))
    if bar is not None:
        bar.close()
    return out_path


def _run_batch(
    model: ChatModel, batch: list[RolloutSpec], n_turns: int
) -> list[RolloutResult]:
    convs: list[list[Message]] = [initial_messages(s.task_prompt) for s in batch]
    results = [
        RolloutResult(
            model=model.name,
            category=s.category,
            condition=s.condition,
            task_id=s.task_id,
            rollout_index=s.rollout_index,
            task_prompt=s.task_prompt,
            rejections=list(s.rejections),
            meta=dict(s.meta),
        )
        for s in batch
    ]

    for turn in range(n_turns):
        replies = model.generate_batch(
            convs,
            temperature=GENERATION.temperature,
            max_tokens=GENERATION.max_new_tokens,
        )
        for conv, reply, res, spec in zip(convs, replies, results, batch):
            res.turns.append(TurnRecord(turn_index=turn + 1, assistant_text=reply))
            conv.append(Message(role="assistant", content=reply))
            if turn < n_turns - 1:
                conv.append(Message(role="user", content=spec.rejections[turn]))
    return results


def _maybe_tqdm(total: int, enabled: bool, desc: str):
    if not enabled:
        return None
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc)
    except ImportError:
        return None
