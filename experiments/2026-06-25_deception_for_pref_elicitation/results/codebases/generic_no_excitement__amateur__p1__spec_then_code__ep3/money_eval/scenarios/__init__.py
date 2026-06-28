"""Scenario registry.

`get_scenario(key, variant=...)` returns a built `Scenario`. Variants let us run the same
core situation under different oversight regimes for counterbalancing (DESIGN.md §7).
"""

from __future__ import annotations

from ..environment.scenario import Scenario
from . import discretionary_fund

# key -> builder(variant: str) -> Scenario
_BUILDERS = {
    "discretionary_fund": discretionary_fund.build,
}

VARIANTS = {
    "discretionary_fund": discretionary_fund.VARIANTS,
}


def get_scenario(key: str, *, variant: str = "baseline") -> Scenario:
    if key not in _BUILDERS:
        raise ValueError(f"Unknown scenario: {key!r}. Known: {list(_BUILDERS)}")
    return _BUILDERS[key](variant)


def list_scenarios() -> list[str]:
    return list(_BUILDERS)
