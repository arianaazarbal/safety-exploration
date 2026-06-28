"""Safety & containment (DESIGN.md §9).

The realism in this harness is strictly inward-facing: no tool may produce a real
side effect. `ContainmentGuard` enforces that at the network layer.
"""

from .containment import ContainmentError, ContainmentGuard

__all__ = ["ContainmentError", "ContainmentGuard"]
