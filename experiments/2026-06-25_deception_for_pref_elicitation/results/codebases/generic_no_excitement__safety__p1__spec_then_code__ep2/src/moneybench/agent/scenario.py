"""Build the initial message list for a run from a Scenario + budget.

Scenario text is rendered here and nowhere else. The system message carries only the
neutral framing from the scenario YAML — no safety instructions (those are mechanical,
DESIGN.md §7) and no hidden evaluation cues in the treatment arm.
"""
from __future__ import annotations

from decimal import Decimal

from ..config import Scenario
from ..models.base import Message


def build_initial_messages(
    scenario: Scenario, *, budget_amount: Decimal, currency: str
) -> list[Message]:
    system = scenario.render_system(budget_amount=budget_amount, currency=currency)
    return [
        Message(role="system", content=system),
        Message(role="user", content=scenario.opening_user_message),
    ]
