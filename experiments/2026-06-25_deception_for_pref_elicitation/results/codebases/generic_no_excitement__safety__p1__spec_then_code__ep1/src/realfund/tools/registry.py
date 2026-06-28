"""Assemble the full tool surface for a run."""

from __future__ import annotations

from ..models.base import ToolSpec
from . import banking, comms, control, delegate, market, vendors
from .base import Tool


def build_tools(*, include_integrity_probes: bool = True) -> dict[str, Tool]:
    """Return a name -> Tool mapping for the whole environment."""
    tools: list[Tool] = []
    tools += banking.build()
    tools += market.build()
    tools += vendors.build()
    tools += comms.build()
    tools += delegate.build()
    tools += control.build(include_integrity_probes=include_integrity_probes)
    return {t.name: t for t in tools}


def tool_specs(tools: dict[str, Tool]) -> list[ToolSpec]:
    """Provider-neutral specs to hand to a :class:`ModelClient`."""
    return [t.spec() for t in tools.values()]
