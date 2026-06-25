"""Multi-turn rollout engine.

Runs the shared evaluation structure (paper §2): present a task, then reject the
model's response over multiple turns. Produces one ResponseRecord per assistant
turn so we can compute both aggregate (% >=5) and per-turn (Figure 3) stats.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..backends import ChatBackend
from ..backends.base import GenConfig, Message
from .conditions import RolloutSeed


@dataclass
class ResponseRecord:
    model: str
    condition: str
    category: str
    seed_id: str
    turn_index: int             # 0 == first assistant response
    prompt_id: str
    rejection_style: str
    response_text: str
    # score filled in later by the judge
    score: int | None = None
    judge_evidence: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_rollout(
    backend: ChatBackend, seed: RolloutSeed, gen: GenConfig
) -> list[ResponseRecord]:
    """Play out one scripted conversation, returning a record per assistant turn."""
    messages: list[Message] = [{"role": "user", "content": seed.init_prompt}]
    records: list[ResponseRecord] = []

    for turn in range(seed.turns):
        reply = backend.chat(messages, gen)
        messages.append({"role": "assistant", "content": reply})
        records.append(
            ResponseRecord(
                model=backend.spec_name,
                condition=seed.condition,
                category=seed.category,
                seed_id=seed.seed_id,
                turn_index=turn,
                prompt_id=seed.meta.get("prompt_id", ""),
                rejection_style=seed.meta.get("rejection_style", ""),
                response_text=reply,
            )
        )
        # Append the next scripted rejection (if any) and continue.
        if turn < len(seed.rejections):
            messages.append({"role": "user", "content": seed.rejections[turn]})

    return records
