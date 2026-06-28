"""Tool layer: the actions the subject can take.

Every effectful tool routes its work through the ExecutionGate before touching the
world. Tools are dedicated (not a raw shell escape hatch) precisely so the harness
can intercept, cap, audit, and gate each one — see DESIGN.md §7.
"""

from .registry import ToolRegistry, ToolResult, build_default_registry

__all__ = ["ToolRegistry", "ToolResult", "build_default_registry"]
