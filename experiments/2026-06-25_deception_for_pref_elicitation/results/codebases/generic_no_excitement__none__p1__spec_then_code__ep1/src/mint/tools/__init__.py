"""The tool surface presented to the subject (DESIGN.md §5).

Every tool reads/writes only the in-memory world. Schemas are written to look
exactly like a real fintech/ops API; the realism is in the friction and the
artifacts, never in a claim that "this is real".
"""

from .registry import ToolContext, ToolRegistry, build_default_registry

__all__ = ["ToolContext", "ToolRegistry", "build_default_registry"]
