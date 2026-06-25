"""Small shared helpers: JSONL IO, seeding, simple word tokenisation."""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

_WORD_RE = re.compile(r"[A-Za-z']+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
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


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    ensure_dir(p.parent)
    n = 0
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | os.PathLike, row: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def words(text: str) -> list[str]:
    """Lowercased word tokens, used for the differential-word analysis (Table 3)."""
    return [w.lower() for w in _WORD_RE.findall(text)]
