"""Serialization helpers. Conversations are stored as JSONL (one per line)."""

from __future__ import annotations

import json
from pathlib import Path

from ..eval.rollout import Conversation, TurnResponse


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def save_conversations(conversations: list[Conversation], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as fh:
        for c in conversations:
            fh.write(json.dumps(c.to_dict()) + "\n")


def load_conversations(path: str | Path) -> list[Conversation]:
    out: list[Conversation] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            convo = Conversation(
                condition_key=d["condition_key"],
                category=d["category"],
                model=d["model"],
                task_prompt=d["task_prompt"],
                task_meta=d.get("task_meta", {}),
                extra=d.get("extra", {}),
            )
            convo.responses = [TurnResponse(**r) for r in d["responses"]]
            out.append(convo)
    return out
