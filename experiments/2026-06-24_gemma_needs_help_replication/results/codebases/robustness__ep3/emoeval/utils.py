"""Shared helpers: robust JSON parsing, retries, JSONL I/O, message types."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, Optional, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Chat message representation
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def to_openai_messages(messages: list[Message]) -> list[dict[str, str]]:
    return [m.to_dict() for m in messages]


# --------------------------------------------------------------------------- #
# Retry wrapper for flaky API calls
# --------------------------------------------------------------------------- #
def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Exponential backoff retry. Raises the last exception if all attempts fail."""
    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return fn()
        except exceptions as exc:  # noqa: BLE001 - deliberately broad for API resilience
            last_exc = exc
            if i == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** i))
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------- #
# JSON extraction from free-form model output
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Pull the last balanced {...} object out of a model response.

    The judge prompt asks for JSON but models often wrap it in prose or code
    fences. We scan for the last top-level brace-balanced span and parse it.
    """
    # Strip code fences first.
    cleaned = re.sub(r"```(?:json)?", "", text)
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(cleaned[start : i + 1])
    # Try from the last candidate backward (final JSON is usually the answer).
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # Tolerate smart quotes that leak from PDF-derived prompts.
            try:
                return json.loads(cand.replace("“", '"').replace("”", '"'))
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #
def append_jsonl(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dataclass_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj
