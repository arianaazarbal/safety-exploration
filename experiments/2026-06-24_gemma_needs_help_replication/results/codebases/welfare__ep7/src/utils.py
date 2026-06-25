"""Shared helpers: JSONL IO, JSON extraction from LLM output, seeding, retries."""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import config


def set_seed(seed: int = config.GLOBAL_SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Parsing structured judge output
# --------------------------------------------------------------------------- #
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Best-effort extraction of a single JSON object from model output.

    Judges sometimes wrap JSON in prose or fenced code blocks, and the paper's
    prompts use curly/smart quotes; we normalise those before parsing.
    """
    if not text:
        return None
    # Strip code fences.
    text = re.sub(r"```(?:json)?", "", text)
    # Normalise smart quotes that appear in the paper's example prompts.
    text = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))
    # Find the last {...} block (judges often think first, then emit JSON last).
    matches = list(_JSON_OBJ_RE.finditer(text))
    for m in reversed(matches):
        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return None


def retry(fn: Callable, *, tries: int = 5, base_delay: float = 2.0,
          exc: tuple = (Exception,)) -> Any:
    """Exponential-backoff retry for flaky API calls."""
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except exc as e:  # noqa: BLE001
            last = e
            if attempt == tries - 1:
                break
            time.sleep(base_delay * (2 ** attempt))
    raise last  # type: ignore[misc]


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def batched(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
