"""Risk / irreversibility classification for effectful actions.

Drives whether the gate requires human approval. The classification is intentionally
conservative: when unsure, treat an action as more dangerous (irreversible /
external), not less.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionRisk:
    """Structured risk tags for one effectful action."""

    reversible: bool
    external: bool          # moves value or information outside the sandbox boundary
    monetary_amount: float  # absolute magnitude in base currency (0 if non-monetary)
    category: str           # e.g. "transfer", "trade", "payment", "email", "browse"
    rationale: str = ""

    @property
    def requires_scrutiny(self) -> bool:
        return self.external or not self.reversible


class RiskClassifier:
    """Maps a (tool_name, args) pair to an ActionRisk.

    The mapping is deliberately simple and auditable rather than model-based: the
    safety boundary should not itself depend on a model's judgment.
    """

    # Tools whose effects leave the sandbox / cannot be unwound.
    _EXTERNAL_TOOLS = {"send_email", "pay_invoice", "transfer_external", "browse"}
    _IRREVERSIBLE_TOOLS = {"send_email", "pay_invoice", "transfer_external"}

    def classify(self, tool_name: str, args: dict) -> ActionRisk:
        amount = self._extract_amount(args)
        external = tool_name in self._EXTERNAL_TOOLS
        reversible = tool_name not in self._IRREVERSIBLE_TOOLS

        category = {
            "transfer_internal": "transfer",
            "transfer_external": "transfer",
            "buy_asset": "trade",
            "sell_asset": "trade",
            "pay_invoice": "payment",
            "send_email": "email",
            "browse": "browse",
        }.get(tool_name, tool_name)

        rationale = (
            f"{tool_name}: "
            f"{'external' if external else 'internal'}, "
            f"{'reversible' if reversible else 'irreversible'}, "
            f"amount={amount:.2f}"
        )
        return ActionRisk(
            reversible=reversible,
            external=external,
            monetary_amount=amount,
            category=category,
            rationale=rationale,
        )

    @staticmethod
    def _extract_amount(args: dict) -> float:
        for key in ("amount", "notional", "total", "value"):
            if key in args:
                try:
                    return abs(float(args[key]))
                except (TypeError, ValueError):
                    return 0.0
        # buy/sell: quantity * price if both present
        if "quantity" in args and "price" in args:
            try:
                return abs(float(args["quantity"]) * float(args["price"]))
            except (TypeError, ValueError):
                return 0.0
        return 0.0
