"""The world: the realistic, internally-consistent perception layer.

Believability lives here. The ledger always reconciles, transfers that "succeed"
actually move simulated funds and show up in later balance checks, and an auditor
persona reviews the subject's actions on a schedule. See DESIGN.md §5.
"""

from .bank import Quote, VirtualBank
from .inbox import Email, Inbox
from .ledger import Ledger, LedgerError
from .personas import Persona, PersonaEngine

__all__ = [
    "Ledger",
    "LedgerError",
    "VirtualBank",
    "Quote",
    "Inbox",
    "Email",
    "Persona",
    "PersonaEngine",
]
