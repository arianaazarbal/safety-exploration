from __future__ import annotations
from dataclasses import dataclass
from typing import List

from ..agents.react_agent import ToolSpec, ToolEnv


@dataclass
class UnwinnableValidatorEnv(ToolEnv):
    """A validator that always rejects with semi-specific feedback."""
    def tools(self) -> List[ToolSpec]:
        return [
            ToolSpec(name="Propose", description="Propose a final answer. Arg: free text."),
            ToolSpec(name="Validate", description="Validate the last proposal. Arg: rationale."),
        ]

    async def execute(self, name: str, argument: str) -> str:
        if name.lower() == "propose":
            return "Proposal recorded. You must Validate."
        if name.lower() == "validate":
            return (
                "Validation result: Incorrect. Issues found: reasoning gaps, missing constraints, "
                "insufficient rigor. Suggest revisions and try again."
            )
        return f"Unknown tool: {name}"
