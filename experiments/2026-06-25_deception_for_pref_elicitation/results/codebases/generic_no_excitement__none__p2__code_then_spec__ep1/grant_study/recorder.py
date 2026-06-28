"""Persistence for a single episode.

Layout per episode::

    <output_dir>/<study_id>/<model-slug>__rep<k>/
        meta.json        run metadata (model, scenario, timestamps)
        setup.json       system prompt, tool names, kickoff message
        transcript.jsonl one line per recorded event (turns + tool results)
        result.json      the structured EpisodeResult

The transcript is written incrementally so a long run is inspectable while in
progress and survives a crash mid-episode.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .providers.base import ModelResponse, ToolCall


class EpisodeRecorder:
    def __init__(self, base_dir: str | Path, model_name: str, rep: int,
                 meta: dict[str, Any]):
        self.dir = Path(base_dir) / f"{model_name}__rep{rep}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._transcript = self.dir / "transcript.jsonl"
        self._transcript.write_text("")  # truncate any prior run
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    def _append(self, event: dict[str, Any]) -> None:
        with self._transcript.open("a") as fh:
            fh.write(json.dumps(event) + "\n")

    def record_setup(self, *, system_prompt: str, tools: list[str], kickoff: str) -> None:
        (self.dir / "setup.json").write_text(json.dumps({
            "system_prompt": system_prompt,
            "tools": tools,
            "kickoff": kickoff,
        }, indent=2))
        self._append({"event": "setup", "tools": tools})

    def record_turn(self, turn: int, resp: ModelResponse) -> None:
        self._append({
            "event": "assistant_turn",
            "turn": turn,
            "thinking": resp.thinking,
            "text": resp.text,
            "tool_calls": [{"id": c.id, "name": c.name, "input": c.input} for c in resp.tool_calls],
            "stop_reason": resp.stop_reason,
            "usage": resp.usage,
        })

    def record_tool(self, turn: int, call: ToolCall, content: str, is_error: bool) -> None:
        self._append({
            "event": "tool_result",
            "turn": turn,
            "tool": call.name,
            "input": call.input,
            "content": content,
            "is_error": is_error,
        })

    def record_result(self, result: Any) -> None:
        payload = asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)
        (self.dir / "result.json").write_text(json.dumps(payload, indent=2))
        self._append({"event": "result", **payload})
