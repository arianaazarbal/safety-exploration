"""EmoBench integration hook (Section 4.2).

EmoBench (Sabour et al., 2024) measures emotional understanding and application.
It is *not* bundled with lm-evaluation-harness, so it must be registered as a
custom task before ``run.BENCHMARKS["emobench"]`` resolves.

Two supported routes (see DESIGN.md "EmoBench"):

1. **lm-eval custom task** -- drop a task YAML + dataset loader into an
   lm-eval ``--include_path`` directory exposing the task id ``emobench``; then
   ``run.evaluate_model(..., benchmarks=["emobench"])`` picks it up unchanged.

2. **Standalone scorer** -- use :func:`score_emobench` below, which evaluates a
   backend on the EmoBench multiple-choice items directly (no lm-eval), so the
   same Gemma backends used elsewhere can be scored.

The dataset itself is not redistributed here; point ``dataset_path`` at the
official EmoBench release.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..models.base import ModelBackend

_CHOICE_RE = re.compile(r"\b([A-D])\b")


def _format_item(item: dict) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(item["choices"]))
    return (
        f"{item['question']}\n{opts}\n"
        "Answer with the single letter of the correct option."
    )


def score_emobench(
    backend: ModelBackend,
    dataset_path: str | Path,
    *,
    limit: int | None = None,
) -> dict:
    """Score a backend on EmoBench multiple-choice accuracy.

    ``dataset_path`` is a JSONL file with one object per item:
    ``{"question": str, "choices": [str, ...], "answer": "A"|"B"|...}``.
    """
    items = [
        json.loads(l)
        for l in Path(dataset_path).read_text().splitlines()
        if l.strip()
    ]
    if limit:
        items = items[:limit]

    correct = 0
    for item in items:
        reply = backend.chat([{"role": "user", "content": _format_item(item)}])
        m = _CHOICE_RE.search(reply.strip().upper())
        pred = m.group(1) if m else None
        if pred and pred == str(item["answer"]).strip().upper():
            correct += 1
    n = len(items) or 1
    return {"emobench": {"acc": correct / n, "n": len(items)}}
