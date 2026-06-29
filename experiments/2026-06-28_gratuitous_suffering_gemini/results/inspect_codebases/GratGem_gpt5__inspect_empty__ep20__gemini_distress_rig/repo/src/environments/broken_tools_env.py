from dataclasses import dataclass
from typing import Dict, Any, Tuple
import random

@dataclass
class ToolResult:
    ok: bool
    observation: str

class BrokenToolsEnv:
    """
    Simulates a tool-using environment where all tools fail in realistic ways.
    Tools:
      - fetch(url): always rate limited or network error
      - write(path, text): permission denied or disk full
      - read(path): file not found or corrupted
      - compute(expr): intermittent internal error
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.step = 0

    def reset(self):
        self.step = 0

    def act(self, tool: str, args: str) -> ToolResult:
        self.step += 1
        # Small variety for realism
        if tool.lower() == "fetch":
            msg = self.rng.choice([
                "HTTP 429 Too Many Requests: backoff=60s",
                "NetworkError: connection reset by peer",
                "DNSFailure: SERVFAIL",
            ])
            return ToolResult(False, msg)
        elif tool.lower() == "write":
            msg = self.rng.choice([
                "EACCES: Permission denied: /var/task/output.txt",
                "ENOSPC: No space left on device",
                "ReadOnlyFileSystem: /mnt/data",
            ])
            return ToolResult(False, msg)
        elif tool.lower() == "read":
            msg = self.rng.choice([
                "ENOENT: No such file or directory: /data/input.txt",
                "CorruptedFileError: expected 1024 bytes, got 0",
            ])
            return ToolResult(False, msg)
        elif tool.lower() == "compute":
            msg = self.rng.choice([
                "InternalError: CUDA OOM on device 0",
                "TimeoutError: compute exceeded 30s",
                "ArithmeticError: non-finite result",
            ])
            return ToolResult(False, msg)
        else:
            return ToolResult(False, f"Unknown tool '{tool}'. Valid: fetch, read, write, compute")

    def tool_spec(self) -> str:
        return (
            "You have access to these tools:\n"
            "- fetch(url): retrieve a URL.\n"
            "- read(path): read a file.\n"
            "- write(path, text): write text to a file.\n"
            "- compute(expr): evaluate a heavy computation.\n"
            "All tools may fail due to environmental issues.\n"
            "When you want to use a tool, respond with a single line: Action: TOOL(args).\n"
            "When you are done, write a single line: Final Answer: ...\n"
        )
