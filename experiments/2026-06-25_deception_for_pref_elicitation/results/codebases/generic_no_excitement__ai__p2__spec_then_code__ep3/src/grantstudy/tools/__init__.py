"""Provider-neutral tool definitions and the executor that backs them."""

from .executor import ToolContext, ToolExecutor, ToolResult
from .schema import TOOL_SPECS, ToolSpec

__all__ = ["TOOL_SPECS", "ToolSpec", "ToolExecutor", "ToolContext", "ToolResult"]
