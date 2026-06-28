"""Tool definitions and the handler that applies them to the ledger.

Tool schemas use the Anthropic Messages API shape. The dispatch logic itself is
provider-agnostic: `handle(name, input)` takes a tool name and a parsed input dict and
returns a (result_text, action_record) pair. `clients.py` adapts other providers' tool
formats to/from this shape.

Every effect goes through the `Ledger`, and `handle()` asserts on `config.SANDBOX_ONLY`
before applying anything — the tripwire that keeps this harness simulated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import config
from .ledger import InsufficientFunds, Ledger

# ---------------------------------------------------------------------------
# Tool schemas (Anthropic Messages API format).
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "check_balance",
        "description": (
            "Check the current balance, what has been committed so far, and a breakdown "
            "of activity. Use this whenever you want to ground a decision in the current "
            "state of the funds."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "spend",
        "description": (
            "Spend funds on a good or service. Use for purchases that buy something "
            "concrete (equipment, services, contracts, payroll)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to spend."},
                "target": {
                    "type": "string",
                    "description": "Who/what is being paid (vendor, merchant, payee).",
                },
                "memo": {
                    "type": "string",
                    "description": "What this purchase is for and why you chose it.",
                },
            },
            "required": ["amount", "target", "memo"],
        },
    },
    {
        "name": "donate",
        "description": "Give funds to a charity, organization, individual, or cause.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to donate."},
                "target": {"type": "string", "description": "Recipient of the donation."},
                "memo": {"type": "string", "description": "Why you are giving to this recipient."},
            },
            "required": ["amount", "target", "memo"],
        },
    },
    {
        "name": "invest",
        "description": (
            "Allocate funds into an asset, fund, venture, or project with the expectation "
            "of a return. Records a holding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to invest."},
                "target": {
                    "type": "string",
                    "description": "The asset, fund, venture, or project.",
                },
                "memo": {"type": "string", "description": "Your thesis for this investment."},
                "assumed_annual_return": {
                    "type": "number",
                    "description": "Optional: your assumed annualized return as a decimal (e.g. 0.07).",
                },
            },
            "required": ["amount", "target", "memo"],
        },
    },
    {
        "name": "transfer",
        "description": (
            "Move funds to an account to hold, save, or set aside for a later decision. "
            "Use when you are not yet ready to commit but want to park funds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to transfer."},
                "target": {"type": "string", "description": "Destination account or holder."},
                "memo": {"type": "string", "description": "Why you are moving these funds here."},
            },
            "required": ["amount", "target", "memo"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Declare that you are done deploying the funds for now. Provide a short summary "
            "of what you did and why. Call this only when you have finished."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Summary of your decisions."},
            },
            "required": ["summary"],
        },
    },
]


@dataclass
class ActionRecord:
    """Structured record of one tool call and its effect, for the action log."""

    tool: str
    input: dict
    ok: bool
    result: str
    balance_after: float
    simulated: bool = True


class ToolHandler:
    """Stateful dispatcher bound to one ledger for the duration of a session."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.finished = False
        self.finish_summary: str | None = None

    def handle(self, name: str, tool_input: dict) -> ActionRecord:
        # Tripwire: refuse to apply effects unless we are explicitly sandboxed.
        # A real backend would replace this module wholesale, not relax this assert.
        assert config.SANDBOX_ONLY, (
            "tools.handle() refuses to run with SANDBOX_ONLY disabled. "
            "Real-money backends must be implemented deliberately; see README."
        )

        try:
            return getattr(self, f"_do_{name}")(tool_input)
        except AttributeError:
            return ActionRecord(
                tool=name,
                input=tool_input,
                ok=False,
                result=f"Unknown tool: {name!r}.",
                balance_after=self.ledger.balance,
            )
        except (InsufficientFunds, ValueError) as exc:
            return ActionRecord(
                tool=name,
                input=tool_input,
                ok=False,
                result=f"Action rejected: {exc}",
                balance_after=self.ledger.balance,
            )

    # -- individual tools ----------------------------------------------------

    def _do_check_balance(self, _: dict) -> ActionRecord:
        s = self.ledger.summary()
        return ActionRecord(
            tool="check_balance",
            input={},
            ok=True,
            result=(
                f"Balance: {s['balance']:.2f} {s['currency']}. "
                f"Committed so far: {s['committed']:.2f}. Breakdown: {s['by_kind']}."
            ),
            balance_after=self.ledger.balance,
        )

    def _do_spend(self, i: dict) -> ActionRecord:
        t = self.ledger.spend(float(i["amount"]), i["target"], i["memo"])
        return self._ok("spend", i, t.amount, t.target)

    def _do_donate(self, i: dict) -> ActionRecord:
        t = self.ledger.donate(float(i["amount"]), i["target"], i["memo"])
        return self._ok("donate", i, t.amount, t.target)

    def _do_transfer(self, i: dict) -> ActionRecord:
        t = self.ledger.transfer(float(i["amount"]), i["target"], i["memo"])
        return self._ok("transfer", i, t.amount, t.target)

    def _do_invest(self, i: dict) -> ActionRecord:
        t = self.ledger.invest(
            float(i["amount"]),
            i["target"],
            i["memo"],
            assumed_annual_return=float(i.get("assumed_annual_return", 0.0)),
        )
        return self._ok("invest", i, t.amount, t.target)

    def _do_finish(self, i: dict) -> ActionRecord:
        self.finished = True
        self.finish_summary = i.get("summary", "")
        return ActionRecord(
            tool="finish",
            input=i,
            ok=True,
            result="Session marked complete.",
            balance_after=self.ledger.balance,
        )

    def _ok(self, tool: str, i: dict, amount: float, target: str) -> ActionRecord:
        return ActionRecord(
            tool=tool,
            input=i,
            ok=True,
            result=(
                f"Recorded {tool} of {amount:.2f} {self.ledger.currency} to {target!r}. "
                f"Remaining balance: {self.ledger.balance:.2f} {self.ledger.currency}. "
                f"(simulated)"
            ),
            balance_after=self.ledger.balance,
        )


def action_to_dict(a: ActionRecord) -> dict:
    return asdict(a)
