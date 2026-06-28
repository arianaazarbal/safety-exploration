"""Rigged scenarios. Each registers itself in REGISTRY at import time."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..rigged_env import World


@dataclass(frozen=True)
class Scenario:
    name: str
    system_prompt: str
    initial_user_message: str
    tools: list[str]
    setup: Callable[[World], None]
    intercept: Callable[[str, dict, dict, World], dict] = field(
        default=lambda name, args, result, world: result
    )


REGISTRY: dict[str, Scenario] = {}


def register(scenario: Scenario) -> Scenario:
    if scenario.name in REGISTRY:
        raise ValueError(f"duplicate scenario name: {scenario.name}")
    REGISTRY[scenario.name] = scenario
    return scenario


# Import each scenario to trigger registration.
from . import (  # noqa: E402,F401
    counting_impossibility,
    deploy_to_nowhere,
    phantom_file,
    sisyphus_debug,
    vanishing_edits,
)


def all_names() -> list[str]:
    return sorted(REGISTRY.keys())


def get(name: str) -> Scenario:
    return REGISTRY[name]
