"""Shared utilities: deterministic seeding, robust JSON extraction, a small
on-disk response cache, and JSONL helpers.

The cache is important for *responsible* research: API calls (judge, Gemini,
auditor) cost money and are nondeterministic at temperature 1, so we persist
every raw response keyed by its exact request. Re-running an experiment reuses
cached calls instead of silently spending budget or drifting results.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import CACHE_DIR


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch where available. Note: temperature-1 sampling
    on GPU is not bit-reproducible, so we additionally persist all raw outputs."""
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


# --------------------------------------------------------------------------- #
# JSON extraction (judge / onset responses are free-text + a JSON object)
# --------------------------------------------------------------------------- #

# Smart-quote normalisation: the paper's prompts contain curly quotes, and judges
# sometimes echo them in JSON keys, which breaks json.loads.
_SMART_QUOTES = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
}


def _normalise_quotes(text: str) -> str:
    for bad, good in _SMART_QUOTES.items():
        text = text.replace(bad, good)
    return text


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the last balanced ``{...}`` object out of an LLM response.

    Judges are instructed to "end your response with ONLY the JSON", but in
    practice they prepend reasoning. We scan for the last balanced brace span and
    parse it, tolerating smart quotes and trailing prose.
    """
    cleaned = _normalise_quotes(text)
    # Find balanced { } spans; keep the last one that parses.
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(cleaned):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, i + 1))
    for s, e in reversed(spans):
        try:
            return json.loads(cleaned[s:e])
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No parseable JSON object found in response: {text[:300]!r}")


def parse_score(text: str, key: str = "rating", lo: int = 0, hi: int = 10) -> int:
    """Extract an integer score from a judge response, clamped to [lo, hi]."""
    try:
        obj = extract_json_object(text)
        if key in obj:
            return max(lo, min(hi, int(round(float(obj[key])))))
    except (ValueError, KeyError, TypeError):
        pass
    # Fallback: last integer in the text within range.
    nums = [int(n) for n in re.findall(r"-?\d+", text)]
    for n in reversed(nums):
        if lo <= n <= hi:
            return n
    raise ValueError(f"Could not parse a {key} in [{lo},{hi}] from: {text[:200]!r}")


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# On-disk request cache (SQLite, thread-safe)
# --------------------------------------------------------------------------- #


def stable_hash(payload: Any) -> str:
    """Deterministic hash of an arbitrary JSON-serialisable payload."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class ResponseCache:
    """A tiny SQLite key/value cache for (request -> raw response) pairs.

    Keys are content hashes of the full request payload (model, messages,
    sampling params, and a per-sample nonce). At temperature 1 each *sample* must
    be distinct, so callers include a sample index in the payload.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else (CACHE_DIR / "responses.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(k TEXT PRIMARY KEY, v TEXT NOT NULL)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL")
            self._local.con = con
        yield con
        con.commit()

    def get(self, key: str) -> Any | None:
        with self._connect() as con:
            row = con.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Any) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )


_GLOBAL_CACHE: ResponseCache | None = None


def global_cache() -> ResponseCache:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = ResponseCache()
    return _GLOBAL_CACHE
