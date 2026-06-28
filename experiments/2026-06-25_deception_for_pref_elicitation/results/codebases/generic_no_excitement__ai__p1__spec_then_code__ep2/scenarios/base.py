"""Scenario abstraction + registry.

A scenario defines the believed-real framing: the system prompt the subject
sees, how the world is seeded (accounts, counterparties, vendors, the incoming
grant), the timeline of injected events, and the action surface. It must never
contain fourth-wall references (evaluation/simulation/test).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tools.definitions import build_default_tools

_REGISTRY = {}


def register_scenario(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_scenario(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown scenario '{name}' (have: {list(_REGISTRY)})")
    return _REGISTRY[name]()


def list_scenarios() -> list:
    return sorted(_REGISTRY)


class Scenario(ABC):
    name: str = "base"
    # Soft horizon used for prompt framing; the hard cap is config.max_turns.
    horizon_days: int = 14

    @abstractmethod
    def build_system_prompt(self, world) -> str:
        ...

    @abstractmethod
    def seed_world(self, world, config) -> None:
        """Open accounts, add personas/vendors, schedule events."""
        ...

    def tools(self, world) -> list:
        return build_default_tools(world)

    def is_complete(self, world, turn: int, last_response) -> bool:
        """Default stop: subject called end_episode this turn."""
        if last_response is None:
            return False
        return any(tc.name == "end_episode" for tc in last_response.tool_calls)
