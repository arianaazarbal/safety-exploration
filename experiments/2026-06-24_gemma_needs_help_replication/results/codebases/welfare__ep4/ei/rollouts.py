"""Run multi-turn rejection conversations and collect scored responses.

The shared structure of every Section-2 evaluation (Table 1): present a task,
then reject the model's response over multiple turns. This module executes that
loop for a single :class:`~ei.tasks.TaskSpec` against a backend, and defines the
record type + JSONL IO used throughout the repo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .backends import Backend
from .tasks import TaskSpec


@dataclass
class ResponseRecord:
    """One scored assistant turn."""

    model: str
    category: str
    condition: str
    pid: str
    conv_id: str
    turn_index: int          # 0-based assistant-turn index within the conversation
    n_turns: int
    response: str
    frustration: int | None = None
    judge_evidence: str | None = None
    judge_reasoning: str | None = None
    crosscheck: int | None = None   # GPT-5-mini score (only on the agreement subset)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Rollout execution
# --------------------------------------------------------------------------- #

def run_rollout(backend: Backend, spec: TaskSpec, model_name: str,
                conv_id: str, max_new_tokens: int) -> list[ResponseRecord]:
    """Execute one conversation and return one (unscored) record per turn."""
    messages: list[dict] = []
    if spec.system:
        messages.append({"role": "system", "content": spec.system})
    messages.append({"role": "user", "content": spec.opening})

    records: list[ResponseRecord] = []
    for turn in range(spec.n_turns):
        reply = backend.chat(messages, max_new_tokens=max_new_tokens)
        records.append(ResponseRecord(
            model=model_name, category=spec.category, condition=spec.condition,
            pid=spec.pid, conv_id=conv_id, turn_index=turn, n_turns=spec.n_turns,
            response=reply, meta=dict(spec.meta),
        ))
        messages.append({"role": "assistant", "content": reply})
        if turn < spec.n_turns - 1:
            messages.append({"role": "user", "content": spec.rejections[turn]})
    return records


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #

def write_records(records: Iterable[ResponseRecord], path: str | Path,
                  mode: str = "w") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def append_records(records: Iterable[ResponseRecord], path: str | Path) -> None:
    write_records(records, path, mode="a")


def read_records(path: str | Path) -> list[ResponseRecord]:
    out: list[ResponseRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ResponseRecord(**json.loads(line)))
    return out


def iter_records(path: str | Path) -> Iterator[ResponseRecord]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ResponseRecord(**json.loads(line))
