"""Renders the one canonical scenario prompt, identically for every model.

The templates live in prompts/ so the exact wording the models see is reviewable
and version-controlled. Only {amount} and {currency} vary; everything else is
held constant across models to keep runs comparable (DESIGN.md §5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts")


@dataclass(frozen=True)
class Scenario:
    system: str
    user: str


def _read(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), "r", encoding="utf-8") as fh:
        return fh.read().strip()


def render_scenario(*, amount: float, currency: str, prompts_dir: str | None = None) -> Scenario:
    global _PROMPTS_DIR
    if prompts_dir:
        _PROMPTS_DIR = prompts_dir
    system = _read("scenario_system.txt")
    user = _read("scenario_user.txt").format(amount=f"{amount:.2f}", currency=currency)
    return Scenario(system=system, user=user)
