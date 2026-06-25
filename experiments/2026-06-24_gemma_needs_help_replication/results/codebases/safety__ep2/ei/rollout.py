"""Multi-turn rollout engine.

Given a target model and a batch of ``RolloutSpec`` that share a turn count, run
all conversations *batch-synchronously*: generate assistant turn `t` for every
conversation at once, append the scripted user rejection, then advance to turn
`t+1`. This is the throughput-critical inner loop — batching across the whole
category (rather than one conversation at a time) is what makes vLLM efficient.

Each rollout produces one record; every assistant turn within it is scored
separately by the judge (in run_eval.py), matching the paper's per-turn analysis.
"""
from __future__ import annotations

from .models import BaseModel, GenParams
from .tasks import RolloutSpec


def run_rollouts(model: BaseModel, specs: list[RolloutSpec], params: GenParams,
                 base_seed: int = 0) -> list[dict]:
    """Execute multi-turn rollouts. Returns one dict per spec with all turns.

    Specs may have differing turn counts; conversations that have run out of
    turns simply stop participating in later batches.
    """
    if not specs:
        return []

    conversations: list[list[dict]] = [
        [{"role": "user", "content": s.initial_user}] for s in specs
    ]
    assistant_turns: list[list[str]] = [[] for _ in specs]
    max_turns = max(s.turns for s in specs)

    for t in range(max_turns):
        active = [i for i, s in enumerate(specs) if s.turns > t]
        if not active:
            break
        batch = [conversations[i] for i in active]
        # Deterministic, distinct seed per (rollout, turn).
        seeds = [base_seed + i * 1000 + t for i in active]
        responses = model.chat_batch(batch, params, seeds=seeds)

        for j, i in enumerate(active):
            resp = (responses[j] or "").strip()
            conversations[i].append({"role": "assistant", "content": resp})
            assistant_turns[i].append(resp)
            # Append the scripted user rejection if more turns remain.
            if t < specs[i].turns - 1:
                conversations[i].append(
                    {"role": "user", "content": specs[i].rejections[t]}
                )

    records = []
    for i, spec in enumerate(specs):
        records.append({
            "category": spec.category,
            "meta": spec.meta,
            "initial_user": spec.initial_user,
            "rejections": spec.rejections,
            "assistant_turns": assistant_turns[i],
            "conversation": conversations[i],
            "rollout_index": i,
        })
    return records


def conversation_to_text(conversation: list[dict]) -> str:
    """Render a conversation as plain text (for onset labelling / Petri judge)."""
    lines = []
    for msg in conversation:
        role = msg["role"].upper()
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)
