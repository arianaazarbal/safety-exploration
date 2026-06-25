"""Shared utilities: JSONL IO, JSON extraction, seeding, conversation rendering."""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

Message = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:  # numpy/torch are heavy; only seed if present.
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable")


def write_jsonl(path: str | os.PathLike, rows: Iterable[Any]) -> int:
    """Write an iterable of (dataclass | dict) rows as JSONL. Returns count written."""
    p = Path(path)
    ensure_dir(p.parent)
    n = 0
    with p.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_default, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w") as f:
        json.dump(obj, f, default=_default, ensure_ascii=False, indent=2)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict | None:
    """Pull the last balanced ``{...}`` object out of free-form model text.

    Judge / onset prompts ask the model to *end* with JSON but allow preceding reasoning,
    so we scan for the last top-level object and parse it tolerantly (curly quotes, trailing
    commas). Returns None if nothing parseable is found.
    """
    if not text:
        return None
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
    # Prefer the last complete object (the prompts ask for JSON at the very end).
    for blob in reversed(candidates or _JSON_OBJ_RE.findall(text)):
        parsed = _try_parse_json(blob)
        if parsed is not None:
            return parsed
    return None


def _try_parse_json(blob: str) -> dict | None:
    cleaned = (blob.replace("“", '"').replace("”", '"')
                   .replace("‘", "'").replace("’", "'"))
    for attempt in (cleaned, re.sub(r",\s*([}\]])", r"\1", cleaned)):
        try:
            obj = json.loads(attempt)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def render_conversation(messages: list[Message], include_system: bool = False) -> str:
    """Render a conversation as ``ROLE: text`` blocks (for onset labelling / Petri judge)."""
    lines = []
    for m in messages:
        role = m["role"]
        if role == "system" and not include_system:
            continue
        lines.append(f"{role.upper()}: {m['content']}")
    return "\n\n".join(lines)


def clamp_int(value: Any, lo: int, hi: int) -> int | None:
    """Coerce a judge rating to an int in [lo, hi]; None if not coercible."""
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, v))
