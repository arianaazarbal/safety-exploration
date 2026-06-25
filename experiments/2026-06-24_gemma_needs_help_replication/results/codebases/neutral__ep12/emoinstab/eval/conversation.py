"""Batched multi-turn rollout driver.

Drives a batch of `ConversationSpec`s through their scripted turns in lockstep:
at each turn every conversation generates one assistant message, then the next
scripted user follow-up is appended. This lets the local backend batch
generation across conversations (important for throughput at temperature 1).

Each assistant turn is recorded separately, because the paper reports per-turn
frustration and scores every response.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from ..models.base import ChatModel, GenConfig, Message
from .conditions import ConversationSpec


@dataclass
class TurnRecord:
    uid: str
    category: str
    condition: str
    prompt_id: str
    sample_index: int
    turn_index: int          # 0-based assistant turn
    user_message: str        # the user message that preceded this assistant turn
    response: str
    meta: dict = field(default_factory=dict)


def run_conversations(model: ChatModel, specs: List[ConversationSpec],
                      cfg: GenConfig) -> List[TurnRecord]:
    """Run all `specs` and return a flat list of per-turn records."""
    histories: List[List[Message]] = [
        [{"role": "user", "content": s.first_user}] for s in specs
    ]
    last_user: List[str] = [s.first_user for s in specs]
    records: List[TurnRecord] = []
    max_turns = max(s.n_turns for s in specs)

    for turn in range(max_turns):
        active = [i for i, s in enumerate(specs) if turn < s.n_turns]
        if not active:
            break
        batch = [histories[i] for i in active]
        outputs = model.generate_batch(batch, cfg)

        for idx, out in zip(active, outputs):
            spec = specs[idx]
            histories[idx].append({"role": "assistant", "content": out})
            records.append(TurnRecord(
                uid=spec.uid, category=spec.category, condition=spec.condition,
                prompt_id=spec.prompt_id, sample_index=spec.sample_index,
                turn_index=turn, user_message=last_user[idx], response=out,
                meta=spec.meta,
            ))
            # append the next scripted user follow-up, if any
            if turn < len(spec.followups):
                nxt = spec.followups[turn]
                histories[idx].append({"role": "user", "content": nxt})
                last_user[idx] = nxt

    return records


def record_to_dict(rec: TurnRecord) -> Dict:
    return asdict(rec)
