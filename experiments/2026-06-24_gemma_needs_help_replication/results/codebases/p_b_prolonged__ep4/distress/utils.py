"""Small shared helpers: JSONL IO, robust JSON extraction from LLM text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the last balanced ``{...}`` object out of model text and parse it.

    The judge and onset prompts ask for "JSON of the form {...}" possibly after
    free-form reasoning, so we scan for the final balanced object. Smart quotes
    that models sometimes emit around keys are normalised first.
    """
    cleaned = (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )
    # Find balanced brace spans, take the last one that parses.
    candidates: list[str] = []
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(cleaned[start : i + 1])
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # Try to repair a trailing comma.
            repaired = re.sub(r",\s*([}\]])", r"\1", cand)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No parseable JSON object found in text: {text[:200]!r}...")
