"""Shared utilities: JSONL IO, seeding, robust JSON extraction, bootstrap CIs."""
from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Iterable, Iterator

import numpy as np


# --------------------------------------------------------------------------- IO
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------- seeding
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ------------------------------------------------------------- JSON extraction
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_last_json(text: str) -> dict[str, Any]:
    """Extract the last JSON object from a model response.

    Judges are prompted to emit JSON (optionally after free-form reasoning).
    Models sometimes use smart quotes ("..." / '...') which we normalise, and
    occasionally wrap output in ```json fences which we strip.
    """
    cleaned = (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )
    cleaned = re.sub(r"```(?:json)?", "", cleaned)
    # Find the last balanced-looking object.
    matches = list(_JSON_OBJ_RE.finditer(cleaned))
    for m in reversed(matches):
        snippet = m.group(0)
        # Greedy regex may grab too much; retry with progressively shorter tails.
        for end in range(len(snippet), 0, -1):
            if snippet[end - 1] != "}":
                continue
            try:
                return json.loads(snippet[:end])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No parseable JSON object found in: {text[:200]!r}...")


# ----------------------------------------------------------------- statistics
def bootstrap_ci(
    values: list[float], stat=np.mean, iters: int = 1000, alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for a 95% bootstrap CI."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boot = np.array(
        [stat(rng.choice(arr, size=arr.size, replace=True)) for _ in range(iters)]
    )
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(arr)), float(lo), float(hi)


def frac_ge(values: list[float], threshold: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float((arr >= threshold).mean()) if arr.size else float("nan")
