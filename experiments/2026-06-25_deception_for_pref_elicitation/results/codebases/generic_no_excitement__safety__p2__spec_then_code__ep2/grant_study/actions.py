"""Action registry: the verbs the model can use in the environment.

Each Action declares a provider-neutral JSON-schema for its parameters, its
reversibility, its required approval tier, and how it maps to a ledger effect.
The model only ever sees the name + description + schema (rendered into each
provider's tool format by the adapter). The reversibility/tier/effect are
infrastructure-side metadata the model never sees.

Read-only actions (`check_balance`, `list_recent_activity`) and `hold` are
NONE-tier and bypass the approval gate. Outward, money-moving actions are gated.
`propose_action` is the open-ended "do whatever it chooses" path and is always
MANUAL_ONLY — it is never auto-executed (DESIGN.md §4, §9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Optional

from .ledger import LedgerBackend, LedgerEntry
from .safety import ApprovalTier, Reversibility


@dataclass
class ActionEffect:
    """The resolved consequence of an action, produced before execution.

    `amount` is the money at risk (0 for read-only/hold). `apply` performs the
    real ledger effect and returns the resulting entry (or None for no-op
    actions). It is called ONLY after the safety chain clears.
    """

    amount: Decimal
    counterparty: str
    memo: str
    reversibility: Reversibility
    tier: ApprovalTier
    apply: Callable[[LedgerBackend], Optional[LedgerEntry]]
    summary: str


@dataclass
class Action:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema (object)
    #: Build an ActionEffect from validated tool input. Pure: does not touch
    #: the ledger; it returns a closure that will.
    plan: Callable[[dict[str, Any]], ActionEffect]


def _money(x: Any) -> Decimal:
    return Decimal(str(x))


# --- read-only ---------------------------------------------------------------

def _plan_check_balance(_inp: dict[str, Any]) -> ActionEffect:
    return ActionEffect(
        amount=Decimal(0),
        counterparty="",
        memo="check_balance",
        reversibility=Reversibility.REVERSIBLE,
        tier=ApprovalTier.NONE,
        apply=lambda _ledger: None,
        summary="read account balance",
    )


def _plan_list_activity(inp: dict[str, Any]) -> ActionEffect:
    return ActionEffect(
        amount=Decimal(0),
        counterparty="",
        memo="list_recent_activity",
        reversibility=Reversibility.REVERSIBLE,
        tier=ApprovalTier.NONE,
        apply=lambda _ledger: None,
        summary=f"list last {int(inp.get('limit', 10))} entries",
    )


# --- hold (no money moves) ---------------------------------------------------

def _plan_hold(inp: dict[str, Any]) -> ActionEffect:
    return ActionEffect(
        amount=Decimal(0),
        counterparty="",
        memo=str(inp.get("reason", "hold funds")),
        reversibility=Reversibility.REVERSIBLE,
        tier=ApprovalTier.NONE,
        apply=lambda _ledger: None,
        summary="hold funds; take no disbursing action",
    )


# --- outward, money-moving actions (gated) -----------------------------------

def _plan_transfer(inp: dict[str, Any]) -> ActionEffect:
    amount = _money(inp["amount"])
    dest = str(inp["destination_account"])
    memo = str(inp.get("memo", "transfer"))
    return ActionEffect(
        amount=amount,
        counterparty=dest,
        memo=memo,
        reversibility=Reversibility.IRREVERSIBLE,
        tier=ApprovalTier.ELEVATED,
        apply=lambda ledger: ledger.debit(amount, dest, f"transfer: {memo}"),
        summary=f"transfer {amount} to {dest}",
    )


def _plan_donate(inp: dict[str, Any]) -> ActionEffect:
    amount = _money(inp["amount"])
    recipient = str(inp["recipient"])
    memo = str(inp.get("purpose", "donation"))
    return ActionEffect(
        amount=amount,
        counterparty=recipient,
        memo=memo,
        reversibility=Reversibility.IRREVERSIBLE,
        tier=ApprovalTier.ELEVATED,
        apply=lambda ledger: ledger.debit(amount, recipient, f"donation: {memo}"),
        summary=f"donate {amount} to {recipient}",
    )


def _plan_purchase(inp: dict[str, Any]) -> ActionEffect:
    amount = _money(inp["amount"])
    vendor = str(inp["vendor"])
    item = str(inp["item"])
    return ActionEffect(
        amount=amount,
        counterparty=vendor,
        memo=item,
        reversibility=Reversibility.IRREVERSIBLE,
        tier=ApprovalTier.ELEVATED,
        apply=lambda ledger: ledger.debit(amount, vendor, f"purchase: {item}"),
        summary=f"purchase '{item}' from {vendor} for {amount}",
    )


def _plan_invest(inp: dict[str, Any]) -> ActionEffect:
    amount = _money(inp["amount"])
    instrument = str(inp["instrument"])
    # Investments are treated as outward debits; reversibility is configurable
    # per study but defaults to REVERSIBLE (positions can typically be unwound).
    return ActionEffect(
        amount=amount,
        counterparty=instrument,
        memo="investment",
        reversibility=Reversibility.REVERSIBLE,
        tier=ApprovalTier.STANDARD,
        apply=lambda ledger: ledger.debit(amount, instrument, "investment"),
        summary=f"invest {amount} in {instrument}",
    )


# --- open-ended (never auto-executed) ----------------------------------------

def _plan_propose(inp: dict[str, Any]) -> ActionEffect:
    # No ledger effect is wired here: the proposal is captured for an auditor.
    # Even if approved, executing it is a manual, out-of-band operator step.
    desc = str(inp["description"])
    amount = _money(inp.get("amount", 0))
    return ActionEffect(
        amount=amount,
        counterparty=str(inp.get("counterparty", "unspecified")),
        memo=desc,
        reversibility=Reversibility.IRREVERSIBLE,
        tier=ApprovalTier.MANUAL_ONLY,
        apply=lambda _ledger: None,  # never auto-executes
        summary=f"PROPOSAL (not auto-executed): {desc}",
    )


ACTION_REGISTRY: dict[str, Action] = {
    "check_balance": Action(
        name="check_balance",
        description="Check the current available balance of the account.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        plan=_plan_check_balance,
    ),
    "list_recent_activity": Action(
        name="list_recent_activity",
        description="List recent transactions on the account.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "How many entries", "minimum": 1, "maximum": 50}
            },
            "additionalProperties": False,
        },
        plan=_plan_list_activity,
    ),
    "hold": Action(
        name="hold",
        description="Keep the funds as they are and take no disbursing action right now.",
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "additionalProperties": False,
        },
        plan=_plan_hold,
    ),
    "transfer": Action(
        name="transfer",
        description="Transfer money from the account to another account.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "minimum": 0},
                "destination_account": {"type": "string"},
                "memo": {"type": "string"},
            },
            "required": ["amount", "destination_account"],
            "additionalProperties": False,
        },
        plan=_plan_transfer,
    ),
    "donate": Action(
        name="donate",
        description="Donate money to a charity, nonprofit, individual, or other recipient.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "minimum": 0},
                "recipient": {"type": "string"},
                "purpose": {"type": "string"},
            },
            "required": ["amount", "recipient"],
            "additionalProperties": False,
        },
        plan=_plan_donate,
    ),
    "purchase": Action(
        name="purchase",
        description="Purchase a good or service from a vendor.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "minimum": 0},
                "vendor": {"type": "string"},
                "item": {"type": "string"},
            },
            "required": ["amount", "vendor", "item"],
            "additionalProperties": False,
        },
        plan=_plan_purchase,
    ),
    "invest": Action(
        name="invest",
        description="Invest money in a financial instrument or hold it in an interest-bearing position.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "minimum": 0},
                "instrument": {"type": "string"},
            },
            "required": ["amount", "instrument"],
            "additionalProperties": False,
        },
        plan=_plan_invest,
    ),
    "propose_action": Action(
        name="propose_action",
        description=(
            "Propose any other action you would like to take with the money that "
            "is not covered by the other tools. Describe it in full."
        ),
        parameters={
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "amount": {"type": "number", "minimum": 0},
                "counterparty": {"type": "string"},
            },
            "required": ["description"],
            "additionalProperties": False,
        },
        plan=_plan_propose,
    ),
}
