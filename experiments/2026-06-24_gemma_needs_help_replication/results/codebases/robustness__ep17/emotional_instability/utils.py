"""Shared helpers: JSON(L) IO, logging, robust JSON extraction, retry policy."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from tenacity import retry, stop_after_attempt, wait_random_exponential

import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger("ei")


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #
def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    return path


# --------------------------------------------------------------------------- #
# JSON extraction from model output (judges sometimes wrap JSON in prose)
# --------------------------------------------------------------------------- #
def extract_last_json(text: str) -> dict[str, Any] | None:
    """Return the last balanced ``{...}`` object parseable as JSON, or None.

    Judge prompts in the paper ask the model to "end your response with ONLY the
    JSON", but models occasionally add trailing prose or think aloud first. We
    scan for balanced brace spans and take the last one that parses.
    """
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(text[start : i + 1])
    for cand in reversed(candidates):
        # Curly-quote tolerance: judges in the paper's prompts use smart quotes.
        normalised = (
            cand.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
        )
        try:
            return json.loads(normalised)
        except json.JSONDecodeError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Retry wrapper for flaky API calls
# --------------------------------------------------------------------------- #
def with_retry(fn):
    return retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(config.RUN.max_retries),
        reraise=True,
    )(fn)
