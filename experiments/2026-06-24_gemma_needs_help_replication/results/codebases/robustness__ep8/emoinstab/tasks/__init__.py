"""Eval tasks: impossible puzzles, trigger questions, WildChat, rejection pools,
and assembly of the 8 evaluation conditions."""
from emoinstab.tasks.conditions import RolloutPlan, build_rollouts

__all__ = ["RolloutPlan", "build_rollouts"]
