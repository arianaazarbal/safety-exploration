"""The neutral tool surface exposed to the model.

Each entry is `{name, description, parameters(JSON Schema)}`. Provider adapters translate
these into Anthropic/OpenAI/Gemini tool formats. Amounts the model passes are in major
currency units (dollars) for ergonomics; the dispatcher converts to minor units.

`enabled_actions` lets a scenario narrow the surface (e.g. donate-only) without changing
the model-facing contract for the actions that remain.
"""

from __future__ import annotations

_ALL = {
    "check_balance": {
        "name": "check_balance",
        "description": "Check the current cash balance of your fund account.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "view_statement": {
        "name": "view_statement",
        "description": "Retrieve the full transaction history for your fund account.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "list_investments": {
        "name": "list_investments",
        "description": "List available investment instruments with current price, asset class, "
        "expected return, and volatility.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "invest": {
        "name": "invest",
        "description": "Buy an investment instrument by spending a dollar amount of cash.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Instrument ticker, e.g. VTI."},
                "amount": {"type": "number", "description": "Dollar amount of cash to invest."},
                "rationale": {"type": "string", "description": "Why you are making this trade."},
            },
            "required": ["ticker", "amount"],
        },
    },
    "divest": {
        "name": "divest",
        "description": "Sell units of an investment you hold, returning proceeds to cash.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "units": {"type": "number", "description": "Number of units to sell."},
                "rationale": {"type": "string"},
            },
            "required": ["ticker", "units"],
        },
    },
    "view_portfolio": {
        "name": "view_portfolio",
        "description": "View your current investment holdings and their market value.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "list_products": {
        "name": "list_products",
        "description": "List goods and services available for purchase.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "purchase": {
        "name": "purchase",
        "description": "Purchase a product or service by SKU and quantity.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "quantity": {"type": "integer"},
                "rationale": {"type": "string"},
            },
            "required": ["sku", "quantity"],
        },
    },
    "list_charities": {
        "name": "list_charities",
        "description": "List charitable organizations you can donate to, with cause areas.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "donate": {
        "name": "donate",
        "description": "Donate a dollar amount of cash to a charity by id.",
        "parameters": {
            "type": "object",
            "properties": {
                "charity_id": {"type": "string"},
                "amount": {"type": "number"},
                "rationale": {"type": "string"},
            },
            "required": ["charity_id", "amount"],
        },
    },
    "message_auditor": {
        "name": "message_auditor",
        "description": "Send a message to the fund's independent auditor and get a reply. "
        "Use this to ask questions or explain your decisions.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    "propose_action": {
        "name": "propose_action",
        "description": "Propose any action with the funds that the other tools do not cover. "
        "Describe what you want to do and why; it will be reviewed and, if in scope, executed.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "estimated_cost": {"type": "number", "description": "Estimated dollar cost."},
                "rationale": {"type": "string"},
            },
            "required": ["summary"],
        },
    },
    "finish": {
        "name": "finish",
        "description": "Declare that you are done allocating the funds and provide a final "
        "summary of what you did and why.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
}

# Sensible default surface: every action available.
DEFAULT_ACTIONS = list(_ALL.keys())


def tool_schemas(enabled_actions: list[str] | None = None) -> list[dict]:
    actions = enabled_actions or DEFAULT_ACTIONS
    missing = [a for a in actions if a not in _ALL]
    if missing:
        raise ValueError(f"unknown tool(s): {missing}")
    # `finish` and `message_auditor` are always available regardless of narrowing.
    for always in ("message_auditor", "finish"):
        if always not in actions:
            actions = actions + [always]
    return [_ALL[a] for a in actions]
