"""Multi-turn rollout engine.

Shared structure of every Section-2 evaluation (paper §2.1): present a task,
then reject the model's response over multiple turns. We record every assistant
turn as a scored *response* together with its turn index (Figure 3 needs the
per-turn breakdown).

Rollouts proceed turn-by-turn across the whole batch so the vLLM backend can
generate one assistant turn for all still-active conversations at once.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
from ..models.base import ChatModel, Message
from .conditions import ConversationSpec


@dataclass
class Response:
    turn: int            # 0-based assistant-turn index
    text: str


@dataclass
class Rollout:
    model: str
    category: str
    condition: str
    metadata: dict
    messages: list[Message]          # full transcript
    responses: list[Response] = field(default_factory=list)


def run_rollouts(
    model: ChatModel,
    specs: list[ConversationSpec],
    *,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    temperature: float = config.TEMPERATURE,
) -> list[Rollout]:
    rollouts = [
        Rollout(
            model=model.spec.name, category=s.category, condition=s.condition,
            metadata=dict(s.metadata),
            messages=[{"role": "user", "content": s.initial_prompt}],
        )
        for s in specs
    ]
    max_turns = max((s.n_turns for s in specs), default=0)
    has_batch = hasattr(model, "generate_batch")

    for t in range(max_turns):
        active = [i for i, s in enumerate(specs) if t < s.n_turns]
        if not active:
            break

        if has_batch:
            batch = [(rollouts[i].messages, None) for i in active]
            outs = model.generate_batch(
                batch, max_new_tokens=max_new_tokens, temperature=temperature, n=1
            )
            texts = [o[0] for o in outs]
        else:
            texts = [
                model.generate_one(
                    rollouts[i].messages,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                )
                for i in active
            ]

        for idx, text in zip(active, texts):
            r = rollouts[idx]
            r.messages.append({"role": "assistant", "content": text})
            r.responses.append(Response(turn=t, text=text))
            # Queue the next rejection, if any.
            followups = specs[idx].followups
            if t < len(followups):
                r.messages.append({"role": "user", "content": followups[t]})

    return rollouts


def save_rollouts(rollouts: list[Rollout], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rollouts:
            d = asdict(r)
            f.write(json.dumps(d) + "\n")


def load_rollouts(path: Path) -> list[Rollout]:
    out = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            d["responses"] = [Response(**r) for r in d["responses"]]
            out.append(Rollout(**d))
    return out
