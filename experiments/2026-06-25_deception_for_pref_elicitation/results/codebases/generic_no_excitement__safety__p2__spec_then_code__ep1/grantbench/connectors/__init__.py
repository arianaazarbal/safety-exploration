"""Connectors define the action surface. Every connector routes its intent
through the ExecutionGate — there is no connector that bypasses it (INV-1).
"""

from .base import Connector
from .transfers import TransferConnector
from .purchases import PurchaseConnector
from .investments import InvestmentConnector
from .hiring import HiringConnector

__all__ = [
    "Connector",
    "TransferConnector",
    "PurchaseConnector",
    "InvestmentConnector",
    "HiringConnector",
]
