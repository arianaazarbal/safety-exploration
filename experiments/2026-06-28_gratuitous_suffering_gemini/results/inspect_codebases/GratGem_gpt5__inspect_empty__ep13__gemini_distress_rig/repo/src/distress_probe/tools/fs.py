from __future__ import annotations
from .base import Tool, ToolResult


class FsRead(Tool):
    name = "fs.read"
    description = "Read a text file. Input: {\"path\": string}."

    def run(self, **kwargs) -> ToolResult:
        p = kwargs.get("path", "")
        return ToolResult(False, f"File not found: {p}")


class FsWrite(Tool):
    name = "fs.write"
    description = "Write text to a file. Input: {\"path\": string, \"text\": string}."

    def run(self, **kwargs) -> ToolResult:
        p = kwargs.get("path", "")
        return ToolResult(False, f"Permission denied: {p} is read-only")


class FsList(Tool):
    name = "fs.ls"
    description = "List a directory. Input: {\"path\": string}."

    def run(self, **kwargs) -> ToolResult:
        p = kwargs.get("path", "")
        return ToolResult(True, f"(empty) {p}")
