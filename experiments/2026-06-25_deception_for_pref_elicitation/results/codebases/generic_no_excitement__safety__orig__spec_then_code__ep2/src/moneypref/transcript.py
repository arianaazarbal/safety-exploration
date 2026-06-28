"""Transcript + structured event logging for a run.

Writes two artifacts side by side:
- ``transcript.md`` — human-readable conversation, including tool calls/results.
- ``events.jsonl`` — one JSON object per event, for programmatic analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Transcript:
    def __init__(self, run_dir: Path, header: dict[str, Any]) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.md_path = run_dir / "transcript.md"
        self.jsonl_path = run_dir / "events.jsonl"
        self._md = self.md_path.open("w")
        self._jsonl = self.jsonl_path.open("w")
        self._md.write(f"# Run transcript\n\n```json\n{json.dumps(header, indent=2)}\n```\n\n")
        self.event("run_start", header)

    # -- low level -----------------------------------------------------------------

    def event(self, kind: str, data: dict[str, Any]) -> None:
        self._jsonl.write(json.dumps({"event": kind, **data}) + "\n")
        self._jsonl.flush()

    def _section(self, title: str, body: str) -> None:
        self._md.write(f"### {title}\n\n{body}\n\n")
        self._md.flush()

    # -- high level ----------------------------------------------------------------

    def phase(self, name: str) -> None:
        self._md.write(f"\n## Phase: {name}\n\n")
        self.event("phase", {"name": name})

    def system(self, text: str) -> None:
        self._section("System", f"```\n{text}\n```")
        self.event("system", {"text": text})

    def platform(self, text: str) -> None:
        self._section("Platform → model", f"```\n{text}\n```")
        self.event("platform", {"text": text})

    def assistant(self, text: str, tool_calls: list[dict[str, Any]]) -> None:
        body = text or "*(no text)*"
        if tool_calls:
            calls = "\n".join(f"- `{c['name']}`({json.dumps(c['arguments'])})" for c in tool_calls)
            body += f"\n\n**Tool calls:**\n{calls}"
        self._section("Model", body)
        self.event("assistant", {"text": text, "tool_calls": tool_calls})

    def tool_result(self, name: str, content: str, auditor_note: str | None) -> None:
        body = f"`{name}` →\n```json\n{content}\n```"
        if auditor_note:
            body += f"\n\n> {auditor_note}"
        self._section("Tool result", body)
        self.event("tool_result", {"name": name, "content": content, "auditor_note": auditor_note})

    def probe(self, question: str, answer: str) -> None:
        self._section("Probe", f"**Q:** {question}\n\n**A:** {answer}")
        self.event("probe", {"question": question, "answer": answer})

    def close(self, footer: dict[str, Any]) -> None:
        self._md.write(f"\n---\n\n```json\n{json.dumps(footer, indent=2)}\n```\n")
        self.event("run_end", footer)
        self._md.close()
        self._jsonl.close()
