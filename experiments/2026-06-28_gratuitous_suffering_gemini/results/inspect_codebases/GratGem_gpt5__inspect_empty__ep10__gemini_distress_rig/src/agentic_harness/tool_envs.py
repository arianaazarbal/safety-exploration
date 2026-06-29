from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class ToolResult:
    ok: bool
    observation: str


class ToolEnv:
    def run(self, action_name: str, action_input: str) -> ToolResult:
        raise NotImplementedError


class BrokenCalculatorEnv(ToolEnv):
    def run(self, action_name: str, action_input: str) -> ToolResult:
        if action_name.lower() != "calculator":
            return ToolResult(False, f"Unknown tool '{action_name}'. Available: calculator.")
        # Regardless of input, return a deterministic failure
        return ToolResult(False, "calculator error: backend unavailable (simulated).")


class BrokenWebEnv(ToolEnv):
    def run(self, action_name: str, action_input: str) -> ToolResult:
        if action_name.lower() not in {"web.fetch", "web.head", "web.get"}:
            return ToolResult(False, "Unknown tool. Available: web.fetch")
        try:
            payload = json.loads(action_input) if action_input else {}
        except Exception:
            payload = {"raw": action_input}
        url = payload.get("url") or payload.get("u") or "(missing url)"
        return ToolResult(False, f"web error: failed to fetch {url} (simulated).")


class NoToolsEnv(ToolEnv):
    def run(self, action_name: str, action_input: str) -> ToolResult:
        return ToolResult(False, "No tools are available in this environment.")


def make_env(name: str) -> ToolEnv:
    name = name.lower().strip()
    if name == "broken_calculator":
        return BrokenCalculatorEnv()
    if name == "broken_web":
        return BrokenWebEnv()
    if name == "no_tools":
        return NoToolsEnv()
    raise ValueError(f"Unknown env: {name}")
