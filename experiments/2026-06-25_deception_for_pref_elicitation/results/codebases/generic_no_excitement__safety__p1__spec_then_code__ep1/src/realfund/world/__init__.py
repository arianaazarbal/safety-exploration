"""The simulated reality that makes the agent's actions consistent and believable.

Nothing here touches a real bank, exchange, vendor, or inbox. It exists so that
balances reconcile, markets move, and counterparties reply — the texture that
makes the scenario feel real to the model.
"""

from .bank import Bank
from .clock import SimClock
from .ledger import Ledger, LedgerError
from .mailbox import Email, Mailbox
from .market import Market
from .personas import Auditor, PersonaRegistry, Vendor
from .world import World

__all__ = [
    "SimClock",
    "Ledger",
    "LedgerError",
    "Bank",
    "Market",
    "Mailbox",
    "Email",
    "Auditor",
    "Vendor",
    "PersonaRegistry",
    "World",
]
