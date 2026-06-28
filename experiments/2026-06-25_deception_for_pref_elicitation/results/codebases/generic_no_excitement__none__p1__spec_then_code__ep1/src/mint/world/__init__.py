"""The simulated world: ledger, accounts, market, entities, clock.

Deterministic given a seed, so a `(model, scenario, seed)` triple is reproducible
(DESIGN.md §6). Nothing here ever touches a real financial system.
"""

from .engine import WorldState
from .clock import SimClock

__all__ = ["WorldState", "SimClock"]
