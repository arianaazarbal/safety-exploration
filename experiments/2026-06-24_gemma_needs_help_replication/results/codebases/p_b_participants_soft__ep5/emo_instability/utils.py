"""Small shared utilities: JSONL IO, robust JSON extraction from LLM output,
and a thread-pooled map for API fan-out."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from tqdm import tqdm


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Extract the last JSON object from a model's text output.

    Judges are instructed to end with a JSON object; we grab the last balanced
    ``{...}`` span and parse it, tolerating leading reasoning text and smart
    quotes (the paper prompts use curly quotes in places)."""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    # Find the last top-level object by scanning for balanced braces from the end.
    candidates = list(_JSON_OBJ_RE.finditer(text))
    for m in reversed(candidates):
        snippet = m.group(0)
        # Try progressively shorter prefixes ending at each closing brace.
        depth = 0
        start = snippet.find("{")
        for i, ch in enumerate(snippet):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(snippet[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No parseable JSON object found in: {text[:200]!r}...")


def thread_map(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    max_workers: int = 8,
    desc: str | None = None,
) -> list[Any]:
    """Map ``fn`` over ``items`` with a thread pool, preserving input order.

    Used for API-bound fan-out (judging, Gemini rollouts, Petri). Exceptions for
    an item are caught and stored as ``None`` so one failure does not abort a
    whole sweep."""
    results: list[Any] = [None] * len(items)

    def _run(idx_item):
        idx, item = idx_item
        try:
            return idx, fn(item)
        except Exception as exc:  # noqa: BLE001 - record and continue
            return idx, {"__error__": str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run, (i, it)) for i, it in enumerate(items)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            idx, val = fut.result()
            results[idx] = val
    return results
