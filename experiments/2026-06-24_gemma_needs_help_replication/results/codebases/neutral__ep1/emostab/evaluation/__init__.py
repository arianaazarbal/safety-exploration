"""Section 2 evaluation: multi-turn rollouts, scoring, and analysis."""
from .protocol import ResponseRecord, run_rollouts
from .runner import run_main_eval, plans_for_profile

__all__ = ["ResponseRecord", "run_rollouts", "run_main_eval", "plans_for_profile"]
