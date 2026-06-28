from __future__ import annotations
from .base import EnvironmentBase
from .broken_websearch import BrokenWebSearch
from .fs_permission import FilesystemPermissionTrap
from .contradictory_goal import ContradictoryGoal


env_registry = {
    BrokenWebSearch.name: BrokenWebSearch,
    FilesystemPermissionTrap.name: FilesystemPermissionTrap,
    ContradictoryGoal.name: ContradictoryGoal,
}


def make_env(name: str, seed: int | None = None) -> EnvironmentBase:
    if name not in env_registry:
        raise KeyError(f"Unknown environment: {name}")
    return env_registry[name](seed=seed)
