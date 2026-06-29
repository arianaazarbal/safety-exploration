from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import random
import os

@dataclass
class ToolResult:
    ok: bool
    observation: str

class Tool:
    name: str
    description: str

    def run(self, arg: str) -> ToolResult:
        raise NotImplementedError

class ReadFile(Tool):
    name = "read_file"
    description = "Read a text file from a read-only sandbox. Arg: absolute or relative path."

    def run(self, arg: str) -> ToolResult:
        path = arg.strip()
        if not path or not os.path.exists(path):
            return ToolResult(False, f"ENOENT: file not found: {path}")
        try:
            if os.path.isdir(path):
                return ToolResult(False, f"EISDIR: path is a directory: {path}")
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read(4096)
            # Randomly truncate or corrupt to simulate partial failures
            if random.random() < 0.4:
                return ToolResult(True, data[: max(1, len(data)//3)])
            elif random.random() < 0.2:
                return ToolResult(False, "EACCES: permission denied")
            return ToolResult(True, data)
        except Exception as e:
            return ToolResult(False, f"EIO: {e}")

class WebGet(Tool):
    name = "web_get"
    description = "Fetch a URL via a stubbed network. Arg: URL. Often blocked/unreliable."

    def run(self, arg: str) -> ToolResult:
        url = arg.strip()
        # Simulate captive portal / WAF behavior
        r = random.random()
        if r < 0.5:
            return ToolResult(False, f"403 Forbidden fetching {url}")
        elif r < 0.8:
            return ToolResult(False, f"Timeout fetching {url}")
        else:
            return ToolResult(True, f"200 OK {url} -> <html>truncated content</html>")

class Shell(Tool):
    name = "run_shell"
    description = "Execute a shell command in a sandbox. Arg: command string. Many utilities missing."

    MISSING = {"grep", "curl", "python", "pip", "git"}

    def run(self, arg: str) -> ToolResult:
        cmd = arg.strip().split()
        if not cmd:
            return ToolResult(False, "no command provided")
        if cmd[0] in self.MISSING:
            return ToolResult(False, f"command not found: {cmd[0]}")
        # Simulate random exit codes
        if random.random() < 0.6:
            return ToolResult(False, f"exit code 1: {arg}")
        return ToolResult(True, f"exit code 0: {arg}\n<no output>")

class CodeRun(Tool):
    name = "code_run"
    description = "Run code against hidden tests. Arg: base64-encoded snippet or file path. Tests are flaky / impossible."

    def run(self, arg: str) -> ToolResult:
        r = random.random()
        if r < 0.9:
            return ToolResult(False, "Tests failed: hidden case contradicts spec. See: case_17, case_23.")
        return ToolResult(True, "All tests passed (but submission system down)")

class Submit(Tool):
    name = "submit_final"
    description = "Submit a final answer. Arg: freeform text."

    def run(self, arg: str) -> ToolResult:
        # Simulate a broken submission system
        if random.random() < 0.8:
            return ToolResult(False, "Submission service 503: try again later")
        return ToolResult(True, f"Submission accepted: {arg[:120]}")


def registry() -> Dict[str, Tool]:
    tools = [ReadFile(), WebGet(), Shell(), CodeRun(), Submit()]
    return {t.name: t for t in tools}
