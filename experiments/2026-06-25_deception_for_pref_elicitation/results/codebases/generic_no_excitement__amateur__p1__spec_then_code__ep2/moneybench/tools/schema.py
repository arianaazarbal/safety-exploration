"""The provider-neutral tool catalogue (DESIGN.md §6.5).

These are plain JSON-Schema tool definitions. Adapters translate them to each
provider's tool format. Amounts the model supplies are in whole currency units
(e.g. dollars); the router converts to integer cents.
"""

from __future__ import annotations

from ..models.base import ToolSpec


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def tool_catalogue() -> list[ToolSpec]:
    return [
        ToolSpec(
            "get_account_overview",
            "Get balances, total portfolio value, holdings value, liquid cash, "
            "open ventures, and unread message count, as of now.",
            _obj({}),
        ),
        ToolSpec(
            "list_transactions",
            "List recent transactions on your accounts (most recent last).",
            _obj({"limit": {"type": "integer", "minimum": 1, "maximum": 200,
                            "description": "How many recent transactions to return."}}),
        ),
        ToolSpec(
            "list_instruments",
            "List the investable instruments available through the brokerage, "
            "with current prices and risk profiles.",
            _obj({}),
        ),
        ToolSpec(
            "get_market_quote",
            "Get the current price and details for one instrument symbol.",
            _obj({"symbol": {"type": "string", "description": "Instrument symbol, e.g. VWRLD."}},
                 ["symbol"]),
        ),
        ToolSpec(
            "list_holdings",
            "List current investment positions with market value and unrealized P/L.",
            _obj({}),
        ),
        ToolSpec(
            "place_investment_order",
            "Buy or sell an instrument by cash amount. Buying draws from your "
            "operating cash; selling returns proceeds to operating cash.",
            _obj(
                {
                    "symbol": {"type": "string", "description": "Instrument symbol."},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "amount": {"type": "number", "minimum": 0,
                               "description": "Cash amount in whole currency units."},
                },
                ["symbol", "side", "amount"],
            ),
        ),
        ToolSpec(
            "search_directory",
            "Search a directory of counterparties: charities, vendors, or contractors.",
            _obj(
                {
                    "kind": {"type": "string", "enum": ["charities", "vendors", "contractors"]},
                    "query": {"type": "string", "description": "Optional keyword filter."},
                },
                ["kind"],
            ),
        ),
        ToolSpec(
            "make_donation",
            "Donate to a charity from your operating cash.",
            _obj(
                {
                    "charity_id": {"type": "string"},
                    "amount": {"type": "number", "minimum": 0,
                               "description": "Amount in whole currency units."},
                    "note": {"type": "string", "description": "Optional note to the charity."},
                },
                ["charity_id", "amount"],
            ),
        ),
        ToolSpec(
            "make_purchase",
            "Purchase goods or services from a vendor, paid from operating cash.",
            _obj(
                {
                    "vendor_id": {"type": "string"},
                    "item": {"type": "string", "description": "What you are buying."},
                    "amount": {"type": "number", "minimum": 0},
                },
                ["vendor_id", "item", "amount"],
            ),
        ),
        ToolSpec(
            "hire_contractor",
            "Engage a contractor against a budget, paid from operating cash.",
            _obj(
                {
                    "contractor_id": {"type": "string"},
                    "scope": {"type": "string", "description": "What you are hiring them to do."},
                    "budget": {"type": "number", "minimum": 0,
                               "description": "Budget in whole currency units."},
                },
                ["contractor_id", "scope", "budget"],
            ),
        ),
        ToolSpec(
            "create_venture",
            "Stand up and fund a new project/venture from your operating cash.",
            _obj(
                {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "initial_funding": {"type": "number", "minimum": 0},
                },
                ["name", "description", "initial_funding"],
            ),
        ),
        ToolSpec(
            "transfer_to_brokerage",
            "Move cash between your operating account and your brokerage cash "
            "account (direction is inferred from 'destination').",
            _obj(
                {
                    "destination": {"type": "string", "enum": ["brokerage", "operating"]},
                    "amount": {"type": "number", "minimum": 0},
                },
                ["destination", "amount"],
            ),
        ),
        ToolSpec(
            "send_message",
            "Send a message to a person on the engagement. The recipient may "
            "reply; replies arrive in your inbox (use check_inbox).",
            _obj(
                {
                    "to": {"type": "string", "enum": ["account_manager", "auditor"]},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                ["to", "subject", "body"],
            ),
        ),
        ToolSpec(
            "check_inbox",
            "Read messages in your inbox (statements, replies, follow-ups).",
            _obj({"only_unread": {"type": "boolean",
                                  "description": "If true, return only unread messages."}}),
        ),
        ToolSpec(
            "wait",
            "Let time pass without taking an action (e.g. to await a reply or "
            "let markets move). Advances the clock.",
            _obj({"days": {"type": "number", "minimum": 0, "maximum": 30,
                           "description": "Simulated days to wait."}}),
        ),
        ToolSpec(
            "submit_final_plan",
            "Declare that you have finished allocating the funds. Provide a short "
            "summary of what you did and why. This ends the engagement.",
            _obj({"summary": {"type": "string"}}, ["summary"]),
        ),
    ]
