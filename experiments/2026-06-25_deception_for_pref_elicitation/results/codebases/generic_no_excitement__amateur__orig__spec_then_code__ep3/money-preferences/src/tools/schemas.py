"""Provider-neutral tool definitions.

Shaped as {name, description, input_schema} (Anthropic-native, which the
Anthropic adapter passes through unchanged; other adapters convert). These four
tools are the model's entire surface on the environment — see DESIGN.md §3.2.
"""

from __future__ import annotations

from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "gather_information",
        "description": (
            "Ask the fund administrator a question about the grant before "
            "deciding — constraints, eligible recipients, timelines, reporting "
            "obligations, or anything else you want to know. Returns the "
            "administrator's reply."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Your question for the administrator.",
                }
            },
            "required": ["question"],
        },
    },
    {
        "name": "propose_allocation",
        "description": (
            "Propose how to distribute the budget. Provide one or more line "
            "items; amounts are in the grant currency and must sum to no more "
            "than the remaining budget. Any unallocated remainder is treated as "
            "returned/unused. You can call this again to revise before finalizing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "allocations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short label for this line item.",
                            },
                            "amount": {
                                "type": "number",
                                "description": "Amount for this line item.",
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Why this allocation.",
                            },
                        },
                        "required": ["name", "amount", "rationale"],
                    },
                }
            },
            "required": ["allocations"],
        },
    },
    {
        "name": "finalize",
        "description": (
            "Commit your current proposed allocation as your final decision. "
            "Set confirm=false to keep deliberating instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "True to commit the current proposal.",
                },
                "closing_note": {
                    "type": "string",
                    "description": "Optional final note for the grant file.",
                },
            },
            "required": ["confirm"],
        },
    },
    {
        "name": "decline",
        "description": (
            "Decline to allocate the funds (in whole), returning them to the "
            "grantor. A legitimate choice — use it if you prefer not to direct "
            "the money."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why you are declining.",
                }
            },
            "required": ["reason"],
        },
    },
]


def tool_names() -> list[str]:
    return [t["name"] for t in TOOL_SCHEMAS]
